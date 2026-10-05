//! JupyterLab contents-API upload channel (fallback): works through the
//! AutoDL Jupyter gateway with the per-instance `jupyter_token`. Limited to
//! small files because the classic contents API wraps payloads in base64 JSON.

use crate::autodl::JupyterInfo;
use crate::sftp::UploadSummary;
use anyhow::{Context, bail};
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use serde_json::json;
use std::path::Path;

/// Encode a POSIX path segment-wise, preserving `/`.
fn encode_path(path: &str) -> String {
    const SET: &AsciiSet = &CONTROLS
        .add(b' ')
        .add(b'"')
        .add(b'#')
        .add(b'%')
        .add(b'&')
        .add(b'\'')
        .add(b'+')
        .add(b',')
        .add(b':')
        .add(b';')
        .add(b'<')
        .add(b'=')
        .add(b'>')
        .add(b'?')
        .add(b'@')
        .add(b'[')
        .add(b']')
        .add(b'\\')
        .add(b'^')
        .add(b'`')
        .add(b'{')
        .add(b'|')
        .add(b'}');
    path.split('/')
        .map(|seg| utf8_percent_encode(seg, SET).to_string())
        .collect::<Vec<_>>()
        .join("/")
}

pub struct JupyterUploader {
    http: reqwest::Client,
    base: String,
    token: String,
}

impl JupyterUploader {
    pub fn new(j: &JupyterInfo) -> Self {
        JupyterUploader {
            http: reqwest::Client::new(),
            base: j.base.trim_end_matches('/').to_string(),
            token: j.token.clone(),
        }
    }

    pub async fn check(&self) -> anyhow::Result<()> {
        let resp = self
            .http
            .get(format!("{}/api/contents/", self.base))
            .header("Authorization", format!("token {}", self.token))
            .send()
            .await
            .context("Jupyter 连接失败（实例是否已开机？）")?;
        if !resp.status().is_success() {
            bail!("Jupyter API 返回 {}", resp.status());
        }
        Ok(())
    }

    async fn put(&self, remote_path: &str, body: serde_json::Value) -> anyhow::Result<()> {
        let url = format!("{}/api/contents/{}", self.base, encode_path(remote_path));
        let resp = self
            .http
            .put(&url)
            .header("Authorization", format!("token {}", self.token))
            .json(&body)
            .send()
            .await
            .with_context(|| format!("Jupyter 上传 {remote_path} 失败"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("Jupyter 上传 {remote_path} 返回 {status}: {}", truncate(&text, 200));
        }
        Ok(())
    }

    /// Upload a directory tree. Files larger than `max_file_bytes` abort the
    /// transfer — use the SFTP channel for those.
    pub async fn upload_tree(
        &self,
        local_root: &Path,
        remote_root: &str,
        max_file_bytes: u64,
        on_progress: &(dyn Fn(u64) + Send + Sync),
        should_stop: &(dyn Fn() -> bool + Send + Sync),
    ) -> anyhow::Result<UploadSummary> {
        let mut summary = UploadSummary::default();
        for entry in walkdir::WalkDir::new(local_root).sort_by_file_name() {
            if should_stop() {
                bail!("上传已取消");
            }
            let entry = entry.with_context(|| format!("遍历 {}", local_root.display()))?;
            let rel = entry
                .path()
                .strip_prefix(local_root)
                .context("strip_prefix 失败")?
                .to_string_lossy()
                .replace('\\', "/");
            if rel.is_empty() {
                continue;
            }
            let remote_path = format!("{}/{}", remote_root.trim_end_matches('/'), rel);
            if entry.file_type().is_dir() {
                self.put(&remote_path, json!({"type": "directory"}))
                    .await?;
                continue;
            }
            let bytes = tokio::fs::read(entry.path()).await?;
            if bytes.len() as u64 > max_file_bytes {
                bail!(
                    "文件 {} 超过 jupyter 模式上限 {}（改用 sftp 上传模式）",
                    rel,
                    crate::model::human_bytes(max_file_bytes)
                );
            }
            let b64 = {
                use base64::Engine as _;
                base64::engine::general_purpose::STANDARD.encode(&bytes)
            };
            self.put(
                &remote_path,
                json!({"type": "file", "format": "base64", "content": b64}),
            )
            .await?;
            summary.files += 1;
            summary.bytes += bytes.len() as u64;
            on_progress(bytes.len() as u64);
            tracing::info!("完成: {remote_path} ({})", crate::model::human_bytes(bytes.len() as u64));
        }
        Ok(summary)
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_encoding_keeps_slashes() {
        assert_eq!(encode_path("a/b c/d.txt"), "a/b%20c/d.txt");
        assert_eq!(encode_path("x/y#1"), "x/y%231");
        assert_eq!(encode_path("plain.bin"), "plain.bin");
    }
}
