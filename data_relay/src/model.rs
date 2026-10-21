//! Shared data types: task records, status enums, path-safety and formatting helpers.

use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Browser is sending resumable data chunks.
    Receiving,
    /// Local data is being hashed before delivery.
    Verifying,
    /// Created, waiting to run.
    Queued,
    /// Download script executing.
    Running,
    /// Script finished successfully, waiting for upload.
    DownloadDone,
    /// Data being uploaded to AutoDL.
    Uploading,
    /// Uploaded and verified on the instance.
    Uploaded,
    /// Uploaded (and powered off if configured) — terminal success.
    Done,
    /// Failed (script exit code != 0, upload error, quota exceeded, ...).
    Failed,
    /// Cancelled by the operator.
    Cancelled,
}

impl TaskStatus {
    #[allow(dead_code)] // used by tests / future CLI output
    pub fn as_str(self) -> &'static str {
        match self {
            TaskStatus::Receiving => "receiving",
            TaskStatus::Verifying => "verifying",
            TaskStatus::Queued => "queued",
            TaskStatus::Running => "running",
            TaskStatus::DownloadDone => "download_done",
            TaskStatus::Uploading => "uploading",
            TaskStatus::Uploaded => "uploaded",
            TaskStatus::Done => "done",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadMode {
    /// Instance pulls from the relay over HTTP Range (aria2/curl) — the fast
    /// default; SSH/SFTP carries only control-plane traffic.
    Pull,
    /// SFTP over the instance SSH proxy (push; resumable fallback).
    Sftp,
    /// JupyterLab contents API (push; small files only).
    Jupyter,
}

impl UploadMode {
    pub fn as_str(self) -> &'static str {
        match self {
            UploadMode::Pull => "pull",
            UploadMode::Sftp => "sftp",
            UploadMode::Jupyter => "jupyter",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub name: String,
    pub status: TaskStatus,
    pub created_at: i64,
    #[serde(default)]
    pub started_at: Option<i64>,
    #[serde(default)]
    pub finished_at: Option<i64>,
    /// Script file name inside the task directory.
    pub script: String,
    /// Interpreter used to run the script (e.g. python3 / bash).
    pub interpreter: String,
    /// Directory (relative to the task dir) whose contents are uploaded.
    pub output_dir: String,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub auto_upload: bool,
    pub upload_mode: UploadMode,
    #[serde(default)]
    pub remote_path: Option<String>,
    #[serde(default)]
    pub upload_total_bytes: u64,
    #[serde(default)]
    pub upload_done_bytes: u64,
    /// Bytes sent in this attempt, excluding resumed/skipped files.
    #[serde(default)]
    pub upload_transferred_bytes: u64,
    #[serde(default)]
    pub upload_attempt: u64,
    /// preparing / transferring / verifying; separate from the task lifecycle.
    #[serde(default)]
    pub upload_phase: String,
    /// Per-task token guarding the /pull/{task}/... HTTP channel (pull 模式).
    #[serde(default)]
    pub pull_token: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    /// Absent for legacy script tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_upload: Option<crate::data_upload::DataUpload>,
}

pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Resolve `rel` inside `base`, rejecting any path that escapes the base
/// directory (absolute paths, `..` components, drive prefixes).
pub fn resolve_inside(base: &Path, rel: &str) -> Option<PathBuf> {
    let rel = rel.trim().replace('\\', "/");
    let rel = rel.trim_matches('/');
    if rel.is_empty() {
        return Some(base.to_path_buf());
    }
    let p = Path::new(&rel);
    for comp in p.components() {
        match comp {
            Component::Normal(_) => {}
            _ => return None,
        }
    }
    Some(base.join(p))
}

/// Remote-side variant of [`resolve_inside`]: joins onto a POSIX root and
/// rejects traversal, so cleanup/deletes can never escape the managed root.
pub fn join_remote(root: &str, rel: &str) -> Option<String> {
    let rel = rel.trim().replace('\\', "/");
    let rel = rel.trim_matches('/');
    if rel.is_empty() {
        return Some(root.trim_end_matches('/').to_string());
    }
    for seg in rel.split('/') {
        if seg.is_empty() || seg == "." || seg == ".." {
            return None;
        }
    }
    Some(format!("{}/{}", root.trim_end_matches('/'), rel))
}

pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

/// Recursively sum the size of every file under `dir`.
pub fn dir_size(dir: &Path) -> u64 {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_inside_rejects_traversal() {
        let base = Path::new("/tmp/task");
        assert!(resolve_inside(base, "out").is_some());
        assert!(resolve_inside(base, "a/b/c").is_some());
        assert!(resolve_inside(base, "").is_some());
        // leading slashes are stripped (result still stays inside base)
        assert!(resolve_inside(base, "/abs").is_some());
        assert!(resolve_inside(base, "..").is_none());
        assert!(resolve_inside(base, "../escape").is_none());
        assert!(resolve_inside(base, "a/../../b").is_none());
    }

    #[test]
    fn join_remote_rejects_traversal() {
        assert_eq!(
            join_remote("/root/autodl-tmp/relay_uploads", "task1/out/x.bin").as_deref(),
            Some("/root/autodl-tmp/relay_uploads/task1/out/x.bin")
        );
        assert!(join_remote("/root/x", "../etc").is_none());
        assert!(join_remote("/root/x", "a//b").is_none());
        assert_eq!(join_remote("/root/x", "").as_deref(), Some("/root/x"));
    }

    #[test]
    fn human_bytes_formats() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1024), "1.00 KB");
        assert_eq!(human_bytes(24 * 1024 * 1024 * 1024), "24.00 GB");
    }

    #[test]
    fn status_serde_roundtrip() {
        let s = serde_json::to_string(&TaskStatus::DownloadDone).unwrap();
        assert_eq!(s, "\"download_done\"");
        assert_eq!(UploadMode::Sftp.as_str(), "sftp");
    }
}
