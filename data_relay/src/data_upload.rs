//! Resumable browser ingestion. Only completed, SHA-256 verified payloads enter delivery.
use crate::{
    model::{self, Task, TaskStatus, UploadMode},
    runner::Relay,
    sftp::SftpUploader,
};
use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, sync::Arc};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

pub const CHUNK_SIZE: u64 = 16 * 1024 * 1024;
pub const DATASET_ROOT: &str = "/root/autodl-tmp/dataset";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFile {
    pub name: String,
    pub size: u64,
    /// Browser sample fingerprint prevents accidental resume with a different selection.
    pub fingerprint: String,
    #[serde(default)]
    pub chunks: Vec<Option<String>>,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUpload {
    pub instance_uuid: String,
    pub files: Vec<DataFile>,
}

#[derive(Deserialize)]
pub struct CreateUpload {
    pub name: String,
    pub instance_uuid: String,
    pub upload_mode: UploadMode,
    pub files: Vec<DataFile>,
}

pub fn valid_name(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    !name.is_empty()
        && name.len() <= 240
        && !name.starts_with('.')
        && !name.starts_with('_')
        && !name.ends_with(['.', ' '])
        && !name.ends_with(".rpart")
        && !name.ends_with(".aria2")
        && !name
            .chars()
            .any(|c| c.is_control() || "/\\:*?\"<>|".contains(c))
        && !matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        && !(stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.as_bytes()[3].is_ascii_digit())
}

fn valid_hash(hash: &str) -> bool {
    hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
}

pub fn gate(relay: &Relay, id: &str) -> anyhow::Result<Arc<tokio::sync::Mutex<()>>> {
    if relay.store.get(id).is_none() {
        bail!("任务不存在");
    }
    Ok(relay
        .data_locks
        .lock()
        .unwrap()
        .entry(id.to_string())
        .or_default()
        .clone())
}

pub async fn create(relay: &Arc<Relay>, input: CreateUpload) -> anyhow::Result<Task> {
    if input.files.is_empty() || input.files.len() > 256 {
        bail!("请选择 1–256 个数据文件");
    }
    if input.instance_uuid.trim().is_empty() || input.instance_uuid.len() > 200 {
        bail!("请先选择目标 AutoDL 实例");
    }
    if input.upload_mode == UploadMode::Jupyter {
        bail!("数据直传请选择 pull 或 SFTP");
    }
    if input.upload_mode == UploadMode::Pull && relay.config().autodl.pull_base_url.is_empty() {
        bail!("请先设置实例可访问的中继地址，或选择 SFTP");
    }
    let mut names = HashSet::new();
    let mut files = input.files;
    for f in &mut files {
        if !valid_name(&f.name) || !names.insert(f.name.to_lowercase()) {
            bail!("文件名非法或重复：{}", f.name);
        }
        if f.size > 1024_u64.pow(4) || !valid_hash(&f.fingerprint) {
            bail!("文件大小或指纹非法：{}", f.name);
        }
        f.chunks = vec![None; f.size.div_ceil(CHUNK_SIZE) as usize];
        f.sha256 = None;
    }
    let task = Task {
        id: format!("d_{}_{}", model::now_secs(), uuid::Uuid::new_v4().simple()),
        name: if input.name.trim().is_empty() {
            files[0].name.clone()
        } else {
            input.name.chars().take(200).collect()
        },
        status: TaskStatus::Receiving,
        created_at: model::now_secs(),
        started_at: None,
        finished_at: None,
        script: String::new(),
        interpreter: String::new(),
        output_dir: "out".into(),
        exit_code: None,
        auto_upload: false,
        upload_mode: input.upload_mode,
        remote_path: Some(DATASET_ROOT.into()),
        upload_total_bytes: files.iter().map(|f| f.size).sum(),
        upload_done_bytes: 0,
        upload_transferred_bytes: 0,
        upload_attempt: 0,
        upload_phase: String::new(),
        pull_token: None,
        error: None,
        data_upload: Some(DataUpload {
            instance_uuid: input.instance_uuid,
            files,
        }),
    };
    let dir = relay.store.task_dir(&task.id).join("out");
    let result: anyhow::Result<()> = async {
        tokio::fs::create_dir_all(&dir).await?;
        for f in &task.data_upload.as_ref().unwrap().files {
            // Fill in place with bounded chunk buffers; no second full-size merge copy.
            let file = tokio::fs::File::create(dir.join(&f.name)).await?;
            file.set_len(f.size).await?;
        }
        relay.store.insert(task.clone())?;
        Ok(())
    }.await;
    if let Err(error) = result {
        // This fresh UUID directory belongs only to this failed creation.
        tokio::fs::remove_dir_all(relay.store.task_dir(&task.id)).await.ok();
        return Err(error);
    }
    Ok(task)
}

pub async fn chunk(
    relay: &Relay,
    id: &str,
    file_index: usize,
    index: usize,
    hash: &str,
    bytes: &[u8],
) -> anyhow::Result<()> {
    if !valid_hash(hash) || format!("{:x}", Sha256::digest(bytes)) != hash.to_lowercase() {
        bail!("分块 SHA-256 校验失败");
    }
    let lock = gate(relay, id)?;
    let _guard = lock.lock().await;
    let task = relay.store.get(id).context("任务不存在")?;
    if task.status != TaskStatus::Receiving {
        bail!("任务不在接收状态");
    }
    let file = task
        .data_upload
        .as_ref()
        .context("非数据任务")?
        .files
        .get(file_index)
        .context("文件序号非法")?;
    let saved = file.chunks.get(index).context("分块序号非法")?;
    let offset = index as u64 * CHUNK_SIZE;
    if bytes.len() as u64 != CHUNK_SIZE.min(file.size - offset) {
        bail!("分块长度不匹配");
    }
    if let Some(previous) = saved {
        if previous != &hash.to_lowercase() {
            bail!("已有分块内容不同，请重新创建上传任务");
        }
        return Ok(());
    }
    let path = relay.store.task_dir(id).join("out").join(&file.name);
    let mut output = tokio::fs::OpenOptions::new().write(true).open(path).await?;
    output.seek(std::io::SeekFrom::Start(offset)).await?;
    output.write_all(bytes).await?;
    output.sync_data().await?;
    relay.store.update(id, |t| {
        t.data_upload.as_mut().unwrap().files[file_index].chunks[index] = Some(hash.to_lowercase());
    })?;
    Ok(())
}

pub async fn dispatch(relay: &Arc<Relay>, id: &str) -> anyhow::Result<()> {
    let guard = gate(relay, id)?
        .try_lock_owned()
        .map_err(|_| anyhow::anyhow!("任务正在处理，请稍后重试"))?;
    let task = relay.store.get(id).context("任务不存在")?;
    let data = task.data_upload.as_ref().context("非数据任务")?;
    if matches!(
        task.status,
        TaskStatus::Verifying | TaskStatus::Uploading | TaskStatus::Done
    ) {
        bail!("任务已提交或完成");
    }
    if data
        .files
        .iter()
        .any(|f| f.chunks.iter().any(Option::is_none))
    {
        bail!("还有分块未上传，请重新选择原文件继续上传");
    }
    relay.store.update(id, |t| {
        t.status = TaskStatus::Verifying;
        t.error = None;
        t.finished_at = None;
    })?;
    let relay = relay.clone();
    let id = id.to_string();
    tokio::spawn(async move {
        let _guard = guard;
        let result = async {
            verify_local(&relay, &id).await?;
            relay.clone().run_pipeline(&id).await
        }
        .await;
        if let Err(error) = result {
            relay
                .store
                .update(&id, |t| {
                    t.status = TaskStatus::Failed;
                    t.error = Some(format!("{error:#}"));
                    t.finished_at = Some(model::now_secs());
                })
                .ok();
        }
    });
    Ok(())
}

async fn verify_local(relay: &Relay, id: &str) -> anyhow::Result<()> {
    let task = relay.store.get(id).context("任务不存在")?;
    for (index, file) in task
        .data_upload
        .context("非数据任务")?
        .files
        .iter()
        .enumerate()
    {
        let mut input =
            tokio::fs::File::open(relay.store.task_dir(id).join("out").join(&file.name)).await?;
        if input.metadata().await?.len() != file.size {
            bail!("本地文件长度变化：{}", file.name);
        }
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; CHUNK_SIZE as usize];
        for expected in &file.chunks {
            let n = buffer
                .len()
                .min((file.size - input.stream_position().await?) as usize);
            input.read_exact(&mut buffer[..n]).await?;
            let chunk_hash = format!("{:x}", Sha256::digest(&buffer[..n]));
            if expected.as_deref() != Some(&chunk_hash) {
                bail!("本地分块损坏：{}，请新建任务重传", file.name);
            }
            hasher.update(&buffer[..n]);
        }
        let hash = format!("{:x}", hasher.finalize());
        relay.store.update(id, |t| {
            t.data_upload.as_mut().unwrap().files[index].sha256 = Some(hash)
        })?;
    }
    Ok(())
}

pub fn remote_stage(id: &str) -> String {
    format!("{DATASET_ROOT}/.relay_incoming/{id}")
}

pub async fn remote_check(sftp: &SftpUploader, task: &Task, install: bool) -> anyhow::Result<()> {
    let Some(data) = &task.data_upload else {
        return Ok(());
    };
    let files: Vec<_> = data
        .files
        .iter()
        .map(|f| json!({"name": f.name, "size": f.size, "sha256": f.sha256}))
        .collect();
    let payload = json!({"id": task.id, "files": files, "install": install});
    let command = crate::sftp::python_command(
        include_str!("dataset_upload_remote.py"),
        &payload.to_string(),
    );
    let (code, output) = sftp.exec(&command).await?;
    if code != 0 {
        bail!("远端数据校验失败：{}", output.trim());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, store::TaskStore};

    fn setup() -> (Arc<Relay>, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("relay_ingest_{}", uuid::Uuid::new_v4()));
        let mut cfg = Config::default();
        cfg.server.data_dir = dir.to_string_lossy().into();
        cfg.server.api_token = "test-only-token".into();
        (Relay::new(dir.join("config.toml"), cfg).unwrap(), dir)
    }

    fn input(name: &str, size: u64) -> CreateUpload {
        CreateUpload {
            name: "test".into(),
            instance_uuid: "test-instance".into(),
            upload_mode: UploadMode::Sftp,
            files: vec![DataFile {
                name: name.into(),
                size,
                fingerprint: "a".repeat(64),
                chunks: vec![],
                sha256: None,
            }],
        }
    }

    #[test]
    fn portable_names_cannot_escape_or_collide_with_control_files() {
        for name in [
            "../x",
            "/x",
            "C:\\x",
            "a/b",
            "..",
            ".env",
            "_relay_pull.sh",
            "x\ny",
            "CON.txt",
            "LPT1",
            "a:stream",
            "x.rpart",
            "x.aria2",
            "x.",
            "x ",
        ] {
            assert!(!valid_name(name), "{name}");
        }
        assert!(valid_name("训练数据 01.parquet"));
    }

    #[tokio::test]
    async fn out_of_order_retry_persistence_and_full_hash() {
        let (relay, dir) = setup();
        let task = create(&relay, input("data.parquet", CHUNK_SIZE + 3))
            .await
            .unwrap();
        let tail = b"end";
        let tail_hash = format!("{:x}", Sha256::digest(tail));
        chunk(&relay, &task.id, 0, 1, &tail_hash, tail)
            .await
            .unwrap();
        chunk(&relay, &task.id, 0, 1, &tail_hash, tail)
            .await
            .unwrap();
        assert!(dispatch(&relay, &task.id).await.is_err());
        let other_hash = format!("{:x}", Sha256::digest(b"bad"));
        assert!(chunk(&relay, &task.id, 0, 1, &other_hash, b"bad")
            .await
            .is_err());
        assert!(chunk(&relay, &task.id, 0, 0, &tail_hash, tail)
            .await
            .is_err());
        let store = TaskStore::open(&dir.join("tasks")).unwrap();
        assert_eq!(
            store.get(&task.id).unwrap().data_upload.unwrap().files[0].chunks[1],
            Some(tail_hash)
        );
        let first = vec![7; CHUNK_SIZE as usize];
        let first_hash = format!("{:x}", Sha256::digest(&first));
        chunk(&relay, &task.id, 0, 0, &first_hash, &first)
            .await
            .unwrap();
        verify_local(&relay, &task.id).await.unwrap();
        let mut expected = Sha256::new();
        expected.update(first);
        expected.update(tail);
        assert_eq!(
            relay
                .store
                .get(&task.id)
                .unwrap()
                .data_upload
                .unwrap()
                .files[0]
                .sha256,
            Some(format!("{:x}", expected.finalize()))
        );
        relay
            .store
            .update(&task.id, |t| t.status = TaskStatus::Uploading)
            .unwrap();
        assert_eq!(
            TaskStore::open(&dir.join("tasks"))
                .unwrap()
                .get(&task.id)
                .unwrap()
                .status,
            TaskStatus::Failed
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn rejects_duplicates_invalid_hash_and_busy_mutation() {
        let (relay, dir) = setup();
        let mut request = input("A.jsonl", 1);
        request.files.push(input("a.jsonl", 1).files.remove(0));
        assert!(create(&relay, request).await.is_err());
        let task = create(&relay, input("data.jsonl", 1)).await.unwrap();
        assert!(chunk(&relay, &task.id, 0, 0, &"0".repeat(64), b"x")
            .await
            .is_err());
        relay
            .store
            .update(&task.id, |t| t.status = TaskStatus::Verifying)
            .unwrap();
        let hash = format!("{:x}", Sha256::digest(b"x"));
        assert!(chunk(&relay, &task.id, 0, 0, &hash, b"x").await.is_err());
        assert!(relay.store.remove(&task.id).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn http_auth_chunk_limit_and_range() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;
        let (relay, dir) = setup();
        let app = crate::web::router(relay.clone());
        let response = app
            .clone()
            .oneshot(
                Request::post("/api/data/uploads")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 401);
        let task = create(&relay, input("data.bin", 6)).await.unwrap();
        let uri = format!("/api/data/uploads/{}/files/0/chunks/0", task.id);
        let response = app
            .clone()
            .oneshot(
                Request::put(&uri)
                    .header("authorization", "test-only-token")
                    .body(Body::from(vec![0u8; CHUNK_SIZE as usize + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 413);
        let hash = format!("{:x}", Sha256::digest(b"abcdef"));
        let response = app
            .clone()
            .oneshot(
                Request::put(&uri)
                    .header("authorization", "test-only-token")
                    .header("x-chunk-sha256", hash)
                    .body(Body::from("abcdef"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        relay
            .store
            .update(&task.id, |t| t.pull_token = Some("range-test".into()))
            .unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::get(format!("/pull/{}/data.bin?t=range-test", task.id))
                    .header("range", "bytes=1-3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 206);
        let body = axum::body::to_bytes(response.into_body(), 100)
            .await
            .unwrap();
        assert_eq!(&body[..], b"bcd");
        let response = app
            .oneshot(
                Request::get(format!("/pull/{}/data.bin?t=bad", task.id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 403);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
