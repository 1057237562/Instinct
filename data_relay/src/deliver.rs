//! Deliver local corpus files into an existing remote Instinct checkout.
use crate::{autodl::AutoDl, config::Config, sftp::SftpUploader};
use anyhow::{bail, Context};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::Read,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

const REMOTE_HELPER: &str = include_str!("delivery_remote.py");

fn python_command(script: &str, argument: &str) -> String {
    format!("export PATH=\"/root/miniconda3/bin:/root/miniconda/bin:/opt/conda/bin:$PATH\"; python3 -c {} {}", quote(script), quote(argument))
}

#[derive(Default)]
pub struct Options {
    pub source: PathBuf,
    pub repo: String,
    pub remove_old: Vec<String>,
    pub execute: bool,
    pub init_repo: bool,
    pub branch: String,
    pub clean_cache: bool,
    pub git_bundle: Option<PathBuf>,
    pub skip_git_update: bool,
}

fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

async fn remote(sftp: &SftpUploader, payload: &Value) -> anyhow::Result<String> {
    let command = python_command(REMOTE_HELPER, &payload.to_string());
    let (code, output) = sftp.exec(&command).await?;
    if code != 0 {
        bail!("Remote delivery check failed (exit {code}):\n{output}");
    }
    Ok(output)
}

fn digest(path: &std::path::Path) -> anyhow::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut buf)?;
        if count == 0 {
            break;
        }
        hash.update(&buf[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub async fn inspect_server(cfg: &Config) -> anyhow::Result<()> {
    server_session(cfg, None).await
}

pub async fn run_remote_script(cfg: &Config, script: &std::path::Path) -> anyhow::Result<()> {
    if !script.is_file() {
        bail!("Maintenance script does not exist: {}", script.display());
    }
    server_session(cfg, Some(script)).await
}

async fn server_session(cfg: &Config, remote_script: Option<&std::path::Path>) -> anyhow::Result<()> {
    let client = AutoDl::new(&cfg.autodl)?;
    for instance in client.list().await? {
        println!(
            "Instance {} status={} name={:?}",
            instance.uuid, instance.status, instance.name
        );
    }
    let info = client.ensure_running().await?;
    println!("Inspecting {} status={}", info.uuid, info.status);
    let ssh = info.ssh.context("Instance has no SSH credentials")?;
    let password = if cfg.autodl.ssh_password.is_empty() {
        &ssh.password
    } else {
        &cfg.autodl.ssh_password
    };
    let known_path = PathBuf::from(&cfg.server.data_dir).join("known_hosts.json");
    let mut known: HashMap<String, String> = if known_path.exists() {
        serde_json::from_slice(&std::fs::read(&known_path)?)?
    } else {
        HashMap::new()
    };
    let host_key = format!("{}:{}", ssh.host, ssh.port);
    let (sftp, fingerprint) = SftpUploader::connect(
        &ssh.host,
        ssh.port,
        &cfg.autodl.ssh_user,
        password,
        true,
        known.get(&host_key).cloned(),
    )
    .await?;
    known.insert(host_key, fingerprint);
    if let Some(parent) = known_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(known_path, serde_json::to_vec_pretty(&known)?)?;
    if let Some(path) = remote_script {
        let script = std::fs::read_to_string(path)?;
        let (code, output) = sftp.exec(&python_command(&script, "")).await?;
        println!("{output}");
        if code != 0 { bail!("Remote script failed (exit {code})"); }
        return Ok(());
    }
    let (code, system) = sftp
        .exec("id; pwd; du -h --max-depth=2 /root/autodl-tmp 2>/dev/null")
        .await?;
    println!("Server environment (exit {code}):\n{system}");
    let script = r#"import os,json,subprocess,shutil
from pathlib import Path
found=set()
for base in ['/root/autodl-tmp','/root','/workspace','/data']:
 if not os.path.isdir(base): continue
 print(json.dumps({'root':base,'entries':[{'name':p.name,'directory':p.is_dir()} for p in Path(base).iterdir() if not p.name.startswith('.')]},ensure_ascii=False))
 for root,dirs,files in os.walk(base):
  dirs[:]=[d for d in dirs if not d.startswith('.') and d not in ['node_modules','target','site-packages','miniconda3','miniconda','anaconda3','venv','envs','proc','sys']]
  if Path(root).relative_to(base).parts.__len__()>4: dirs[:]=[]; continue
  if os.path.exists(os.path.join(root,'.git')) or (os.path.isfile(os.path.join(root,'model','model_instinct.py')) and os.path.isdir(os.path.join(root,'trainer'))):
   found.add(root); dirs[:]=[]
for root in sorted(found):
 def git(*args):
  return subprocess.run(['git','-C',root,*args],capture_output=True,text=True).stdout.strip()
 dataset=Path(root)/'dataset'
 items=[]
 if dataset.is_dir():
  for path in sorted(dataset.iterdir()):
   items.append({'name':path.name,'directory':path.is_dir(),'bytes':path.stat().st_size if path.is_file() else None})
 print(json.dumps({'repo':root,'origin':git('remote','get-url','origin'),'branch':git('branch','--show-current'),'status':git('status','--short'),'dataset':items,'free_bytes':shutil.disk_usage(root).free},ensure_ascii=False))
if not found: print('No repository found within depth 4; inspect roots manually.')
"#;
    let (code, output) = sftp.exec(&python_command(script, "")).await?;
    println!("{output}");
    if code != 0 {
        bail!("Server inspection failed (exit {code})");
    }
    Ok(())
}

pub async fn run(cfg: &Config, opts: Options) -> anyhow::Result<()> {
    if opts.repo.is_empty() || !opts.repo.starts_with('/') || !opts.source.is_file() {
        bail!("deliver requires --source FILE and --repo /absolute/Instinct");
    }
    let source = opts.source.canonicalize()?;
    let mut paths = vec![source.clone()];
    let compiler_report = PathBuf::from(format!("{}.report.json", source.display()));
    let corpus_report = source.with_extension("report.json");
    for report in [compiler_report, corpus_report] {
        if report.is_file() && !paths.contains(&report) {
            paths.push(report);
        }
    }
    let mut files = Vec::new();
    for path in &paths {
        println!("Hashing {} ...", path.display());
        let hash_path = path.clone();
        let before = path.metadata()?;
        let hash = tokio::task::spawn_blocking(move || digest(&hash_path)).await??;
        let after = path.metadata()?;
        if before.len() != after.len() || before.modified()? != after.modified()? {
            bail!("Source changed during hashing: {}", path.display());
        }
        files.push(json!({"name": path.file_name().context("Missing filename")?.to_str().context("Filename must be UTF-8")?,
                          "size": after.len(), "sha256": hash}));
    }
    let client = AutoDl::new(&cfg.autodl)?;
    let info = if opts.execute {
        client.ensure_running().await?
    } else {
        client.pick_instance().await?
    };
    println!("Instance {} status={}", info.uuid, info.status);
    if !info.is_running() {
        bail!("Inspection never boots an instance; boot it manually or use --execute");
    }
    let ssh = info.ssh.context("Instance has no SSH credentials")?;
    let password = if cfg.autodl.ssh_password.is_empty() {
        &ssh.password
    } else {
        &cfg.autodl.ssh_password
    };
    let known_path = PathBuf::from(&cfg.server.data_dir).join("known_hosts.json");
    let mut known: HashMap<String, String> = if known_path.exists() {
        serde_json::from_slice(&std::fs::read(&known_path)?)?
    } else {
        HashMap::new()
    };
    let host_key = format!("{}:{}", ssh.host, ssh.port);
    let (sftp, fingerprint) = SftpUploader::connect(
        &ssh.host,
        ssh.port,
        &cfg.autodl.ssh_user,
        password,
        true,
        known.get(&host_key).cloned(),
    )
    .await?;
    // Always pin known keys for this command, including when the WebUI disables strict mode.
    known.insert(host_key, fingerprint);
    if let Some(parent) = known_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&known_path, serde_json::to_vec_pretty(&known)?)?;
    let mut payload = json!({"repo": opts.repo, "remove_old": opts.remove_old, "files": files, "action": "inspect", "init_repo": opts.init_repo, "branch": opts.branch, "clean_cache": opts.clean_cache, "skip_git_update": opts.skip_git_update});
    if let Some(bundle) = &opts.git_bundle {
        payload["git_bundle"] = json!({"name": ".instinct-delivery.bundle", "size": bundle.metadata()?.len(), "sha256": digest(bundle)?});
    }
    let inspection = remote(&sftp, &payload).await?;
    println!("{inspection}");
    if !opts.execute {
        println!("Inspection complete. Use --execute to clean selected old data, git pull --ff-only, upload and verify.");
        return Ok(());
    }
    let plan: Value = serde_json::from_str(inspection.trim())?;
    let dataset = plan["dataset"]
        .as_str()
        .context("Missing remote dataset path")?;
    if let Some(bundle) = &opts.git_bundle {
        let remote_bundle = format!(
            "{}/.instinct-delivery.bundle",
            plan["repo"].as_str().unwrap()
        );
        println!(
            "Uploading Git bundle ({} bytes) ...",
            bundle.metadata()?.len()
        );
        sftp.upload_file(bundle, &remote_bundle, bundle.metadata()?.len(), &|_| {})
            .await?;
    }
    payload["action"] = json!("prepare");
    println!("{}", remote(&sftp, &payload).await?);
    for (path, item) in paths.iter().zip(files.iter()) {
        let name = item["name"].as_str().unwrap();
        if plan["ready_files"]
            .as_array()
            .is_some_and(|ready| ready.iter().any(|value| value.as_str() == Some(name)))
        {
            println!("Skipping already verified temporary upload: {name}");
            continue;
        }
        let size = item["size"].as_u64().unwrap();
        let incoming = format!("{dataset}/{name}.incoming");
        println!(
            "Uploading {} ({} bytes), resumable via .incoming.rpart",
            name, size
        );
        let sent = AtomicU64::new(0);
        let tick = AtomicU64::new(0);
        let progress = |delta| {
            let count = sent.fetch_add(delta, Ordering::Relaxed) + delta;
            let step = count / (128 * 1024 * 1024);
            if tick.swap(step, Ordering::Relaxed) != step {
                println!("{name}: transferred {} MiB this run", count / (1024 * 1024));
            }
        };
        sftp.upload_file(path, &incoming, size, &progress).await?;
    }
    payload["action"] = json!("verify");
    println!("Verifying remote SHA-256 ...");
    println!("{}", remote(&sftp, &payload).await?);
    println!("Delivery complete. Instance left running; this command does not shut down training workloads.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shell_quote_keeps_metacharacters_literal() {
        assert_eq!(quote("a'b;$HOME"), "'a'\\''b;$HOME'");
    }
}
