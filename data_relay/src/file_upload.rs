//! Explicit artifact transfers using the relay's pinned, resumable SFTP channel.
use crate::{autodl::AutoDl, runner::Relay, training};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Deserialize)]
struct Entry {
    source: PathBuf,
    destination: String,
}
#[derive(Deserialize)]
struct Manifest {
    host: String,
    port: u16,
    files: Vec<Entry>,
}

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
fn valid_destination(value: &str) -> bool {
    value.starts_with('/')
        && !value.contains(['\n', '\r', '\0'])
        && value
            .split('/')
            .skip(1)
            .all(|part| !part.is_empty() && part != "." && part != "..")
}
fn hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub async fn run(relay: Arc<Relay>, path: &Path) -> Result<()> {
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(!manifest.files.is_empty(), "empty upload manifest");
    for entry in &manifest.files {
        ensure!(
            entry.source.is_file() && valid_destination(&entry.destination),
            "invalid upload entry"
        );
    }
    let cfg = relay.config();
    let instance = AutoDl::new(&cfg.autodl)?.pick_instance().await?;
    ensure!(
        instance.is_running(),
        "uploads never boot or change GPU mode"
    );
    let ssh = instance.ssh.as_ref().context("missing SSH endpoint")?;
    ensure!(
        ssh.host == manifest.host && ssh.port == manifest.port,
        "instance SSH endpoint differs from manifest"
    );
    // Pin once before dispatching independent file workers.
    let connection = Arc::new(training::connect(&relay, &cfg, &instance).await?);
    let mut lanes = Vec::new();
    if std::env::var("INSTINCT_RELAY_PARALLEL_UPLOAD").as_deref() == Ok("1") {
        for _ in 0..8 {
            lanes.push(Arc::new(training::connect(&relay, &cfg, &instance).await?));
        }
    }
    let lanes = Arc::new(lanes);
    let mut tasks = tokio::task::JoinSet::new();
    let slots = Arc::new(tokio::sync::Semaphore::new(3));
    for entry in manifest.files {
        let connection = connection.clone();
        let slots = slots.clone();
        let lanes = lanes.clone();
        tasks.spawn(async move {
            let _slot = slots.acquire_owned().await?;
            let before = entry.source.metadata()?;
            let source = entry.source.clone();
            let digest = tokio::task::spawn_blocking(move || hash(&source)).await??;
            let after = entry.source.metadata()?;
            ensure!(
                before.len() == after.len() && before.modified()? == after.modified()?,
                "local source changed"
            );
            let destination = &entry.destination;
            let parent = destination
                .rsplit_once('/')
                .context("missing remote parent")?
                .0;
            connection.mkdir_p(parent).await?;
            let remote_hash = |path: String| {
                let connection = connection.clone();
                async move {
                    let (code, output) = connection
                        .exec(&format!("sha256sum -- {}", quote(&path)))
                        .await?;
                    ensure!(code == 0, "remote SHA-256 failed: {}", output.trim());
                    Ok::<_, anyhow::Error>(
                        output.split_whitespace().next().unwrap_or("").to_string(),
                    )
                }
            };
            if connection.metadata_size(destination).await.is_some() {
                ensure!(
                    remote_hash(destination.clone()).await? == digest,
                    "destination already exists with different contents: {destination}"
                );
                println!("VERIFIED existing {destination}");
                return Ok::<_, anyhow::Error>(());
            }
            let incoming = format!("{destination}.incoming");
            let ready = connection.metadata_size(&incoming).await == Some(after.len())
                && remote_hash(incoming.clone()).await? == digest;
            if !ready {
                println!("UPLOAD {destination} bytes={}", after.len());
                let transferred = std::sync::atomic::AtomicU64::new(0);
                let tick = std::sync::atomic::AtomicU64::new(0);
                let progress = |delta| {
                    use std::sync::atomic::Ordering::Relaxed;
                    let count = transferred.fetch_add(delta, Relaxed) + delta;
                    let bucket = count / (128 * 1024 * 1024);
                    if tick.swap(bucket, Relaxed) != bucket {
                        println!("PROGRESS {destination} {} MiB", count / (1024 * 1024));
                    }
                };
                if !lanes.is_empty() && after.len() > 64 * 1024 * 1024 {
                    parallel(
                        &connection,
                        &lanes,
                        &entry.source,
                        &incoming,
                        after.len(),
                        &digest,
                        &progress,
                    )
                    .await?;
                } else {
                    connection
                        .upload_file(&entry.source, &incoming, after.len(), &progress)
                        .await?;
                }
            }
            ensure!(
                remote_hash(incoming.clone()).await? == digest,
                "uploaded SHA-256 mismatch: {destination}"
            );
            // Hard-link installation is atomic and refuses to overwrite a final file.
            let command = format!(
                "ln -- {} {} && rm -- {}",
                quote(&incoming),
                quote(destination),
                quote(&incoming)
            );
            let (code, output) = connection.exec(&command).await?;
            if code != 0 {
                bail!("verified upload retained at {incoming}: {output}");
            }
            println!("VERIFIED {destination} sha256={digest}");
            Ok(())
        });
    }
    // Finish every active transfer even if a sibling fails, retaining resume parts.
    let mut errors = Vec::new();
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result? {
            errors.push(format!("{error:#}"));
        }
    }
    ensure!(errors.is_empty(), "{}", errors.join("\n"));
    Ok(())
}

async fn parallel(
    connection: &Arc<crate::sftp::SftpUploader>,
    lanes: &[Arc<crate::sftp::SftpUploader>],
    source: &Path,
    incoming: &str,
    bytes: u64,
    digest: &str,
    progress: &(dyn Fn(u64) + Send + Sync),
) -> Result<()> {
    let part = format!("{incoming}.rpart");
    let state = format!("{incoming}.ranges.json");
    let prefix = if connection.metadata_size(&state).await.is_some() {
        let (code, text) = connection
            .exec(&format!("cat -- {}", quote(&state)))
            .await?;
        ensure!(code == 0, "cannot read range plan");
        let value: serde_json::Value = serde_json::from_str(&text)?;
        ensure!(
            value["sha256"] == digest && value["bytes"] == bytes,
            "range plan source mismatch"
        );
        value["prefix"].as_u64().context("invalid range prefix")?
    } else {
        let prefix = connection.metadata_size(&part).await.unwrap_or(0);
        ensure!(prefix <= bytes, "range prefix exceeds file size");
        connection
            .write_file(
                &state,
                serde_json::to_string(
                    &serde_json::json!({"sha256":digest,"bytes":bytes,"prefix":prefix}),
                )?
                .as_bytes(),
            )
            .await?;
        prefix
    };
    const SIZE: u64 = 32 * 1024 * 1024;
    let chunks = (bytes - prefix).div_ceil(SIZE);
    let mut jobs = Vec::new();
    // Futures are polled together, with each connection owning disjoint chunks.
    for (lane_index, lane) in lanes.iter().enumerate() {
        let source = source.to_owned();
        let incoming = incoming.to_owned();
        let lane = lane.clone();
        jobs.push(async move {
            for index in (lane_index as u64..chunks).step_by(lanes.len()) {
                let offset = prefix + index * SIZE;
                let length = SIZE.min(bytes - offset);
                let chunk = format!("{incoming}.segment-{index:06}");
                if lane.metadata_size(&chunk).await == Some(length) {
                    continue;
                }
                lane.upload_slice(&source, &chunk, offset, length, progress)
                    .await?;
            }
            Ok::<_, anyhow::Error>(())
        });
    }
    // A small JoinSet uses scoped, borrowed progress through a shared future set.
    let results = futures_util::future::join_all(jobs).await;
    for result in results {
        result?;
    }
    // Assemble only after every bounded segment is complete. Existing prefix is
    // reused; a failed assembly is retried from the recorded prefix.
    let script = r#"import os,sys,subprocess
p,prefix,count=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
with open(p+'.rpart','ab') as output:
 output.truncate(prefix)
 subprocess.run(['cat','--',*[p+'.segment-'+format(i,'06d') for i in range(count)]],stdout=output,check=True)
"#;
    let command = format!(
        "export PATH=\"{}:$PATH\"; python3 -c {} {} {} {}",
        crate::sftp::REMOTE_PYTHON_PATH,
        quote(script),
        quote(incoming),
        prefix,
        chunks
    );
    if chunks > 0 {
        let (code, text) = connection.exec(&command).await?;
        ensure!(code == 0, "range assembly failed: {text}");
    }
    ensure!(
        connection.metadata_size(&part).await == Some(bytes),
        "assembled size mismatch"
    );
    let (code, text) = connection
        .exec(&format!("sha256sum -- {}", quote(&part)))
        .await?;
    ensure!(
        code == 0 && text.split_whitespace().next() == Some(digest),
        "assembled SHA-256 mismatch; segments retained"
    );
    let (code, text) = connection
        .exec(&format!("mv -n -- {} {}", quote(&part), quote(incoming)))
        .await?;
    ensure!(code == 0, "cannot publish assembled upload: {text}");
    let cleanup = format!("rm -- {}.segment-* {}", quote(incoming), quote(&state));
    if chunks > 0 {
        let (code, text) = connection.exec(&cleanup).await?;
        ensure!(code == 0, "segment cleanup failed: {text}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upload_paths_are_explicit_and_quoted() {
        assert!(valid_destination("/autodl-tmp/checkpoints/full.npz"));
        for path in ["relative", "/tmp/../x", "/tmp//x", "/tmp/x\ny"] {
            assert!(!valid_destination(path));
        }
        assert_eq!(quote("a'b"), "'a'\\''b'");
    }
}
