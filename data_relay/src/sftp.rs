//! SFTP upload channel (primary): password auth through the instance SSH
//! proxy, resumable transfers, remote exec for disk queries and cleanup.

use anyhow::{Context, bail};
use russh::client::Handle;
use russh_sftp::client::SftpSession;
use russh_sftp::protocol::OpenFlags;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

const CHUNK: usize = 256 * 1024;

#[derive(Debug, Clone, Default)]
pub struct UploadSummary {
    pub files: u32,
    pub skipped: u32,
    pub bytes: u64,
}

struct ClientHandler {
    strict: bool,
    expected_fingerprint: Option<String>,
    recorded: Arc<std::sync::Mutex<Option<String>>>,
}

#[async_trait::async_trait]
impl russh::client::Handler for ClientHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &russh::keys::key::PublicKey,
    ) -> Result<bool, Self::Error> {
        let fp = server_public_key.fingerprint();
        *self.recorded.lock().unwrap() = Some(fp.clone());
        match (&self.expected_fingerprint, self.strict) {
            (Some(expected), true) => Ok(expected == &fp),
            _ => Ok(true), // accept-and-record (first use)
        }
    }
}

pub struct SftpUploader {
    handle: Handle<ClientHandler>,
    sftp: SftpSession,
}

impl SftpUploader {
    /// Connect with password auth. Returns the uploader and the observed host
    /// key fingerprint (persist it and pass as `expected_fingerprint` with
    /// `strict = true` to pin the host key).
    pub async fn connect(
        host: &str,
        port: u16,
        user: &str,
        password: &str,
        strict: bool,
        expected_fingerprint: Option<String>,
    ) -> anyhow::Result<(Self, String)> {
        let recorded = Arc::new(std::sync::Mutex::new(None));
        let handler = ClientHandler {
            strict,
            expected_fingerprint,
            recorded: recorded.clone(),
        };
        let config = Arc::new(russh::client::Config::default());
        let mut handle = russh::client::connect(config, (host, port), handler)
            .await
            .with_context(|| format!("SSH 连接失败 {host}:{port}"))?;
        if !handle
            .authenticate_password(user, password)
            .await
            .unwrap_or(false)
        {
            bail!("SSH 密码认证失败 ({user}@{host}:{port})");
        }
        let channel = handle
            .channel_open_session()
            .await
            .context("打开 SFTP 会话失败")?;
        let sftp = SftpSession::new(channel.into_stream())
            .await
            .context("SFTP 初始化失败（实例是否已开机？）")?;
        let fingerprint = recorded.lock().unwrap().clone().unwrap_or_default();
        Ok((SftpUploader { handle, sftp }, fingerprint))
    }

    pub async fn exec(&self, cmd: &str) -> anyhow::Result<(i32, String)> {
        let mut channel = self.handle.channel_open_session().await?;
        channel.exec(false, cmd).await?;
        let mut out = Vec::new();
        let mut exit_code = 0;
        while let Some(msg) = channel.wait().await {
            match msg {
                russh::ChannelMsg::Data { data } => out.extend_from_slice(&data),
                russh::ChannelMsg::ExtendedData { data, .. } => out.extend_from_slice(&data),
                russh::ChannelMsg::ExitStatus { exit_status } => exit_code = exit_status as i32,
                russh::ChannelMsg::Eof => {}
                russh::ChannelMsg::Close => break,
                _ => {}
            }
        }
        Ok((exit_code, String::from_utf8_lossy(&out).into_owned()))
    }

    /// (total, free) bytes of the filesystem holding `path`.
    pub async fn fs_usage(&self, path: &str) -> anyhow::Result<(u64, u64)> {
        let st = self
            .sftp
            .fs_info(path)
            .await?
            .ok_or_else(|| anyhow::anyhow!("无法获取 {path} 的文件系统信息"))?;
        Ok((
            st.fragment_size * st.blocks,
            st.fragment_size * st.blocks_avail,
        ))
    }

    pub async fn mkdir_p(&self, remote_dir: &str) -> anyhow::Result<()> {
        let mut acc = String::new();
        for seg in remote_dir.split('/').filter(|s| !s.is_empty()) {
            acc.push('/');
            acc.push_str(seg);
            if self.sftp.create_dir(&acc).await.is_err() {
                // exists (or failed) — verify it is a directory
                if let Ok(m) = self.sftp.metadata(&acc).await {
                    if !m.is_dir() {
                        bail!("{acc} 已存在且不是目录");
                    }
                }
            }
        }
        Ok(())
    }

    /// Write an in-memory control-plane file (manifest, pull script) to the
    /// instance. Small payloads only — bulk data travels the pull channel.
    pub async fn write_file(&self, remote_path: &str, bytes: &[u8]) -> anyhow::Result<()> {
        let mut f = self
            .sftp
            .open_with_flags(
                remote_path,
                OpenFlags::CREATE | OpenFlags::WRITE | OpenFlags::TRUNCATE,
            )
            .await
            .with_context(|| format!("写远端文件 {remote_path} 失败"))?;
        f.write_all(bytes).await?;
        f.shutdown().await?;
        Ok(())
    }

    pub async fn metadata_size(&self, path: &str) -> Option<u64> {
        self.sftp.metadata(path).await.ok().map(|m| m.len())
    }

    /// Recursively upload `local_root` to `remote_root`, skipping files that
    /// already exist remotely with the same size. Partial transfers land in
    /// `<name>.rpart` and resume from where they stopped.
    pub async fn upload_tree(
        &self,
        local_root: &Path,
        remote_root: &str,
        on_progress: &(dyn Fn(u64) + Send + Sync),
        should_stop: &(dyn Fn() -> bool + Send + Sync),
    ) -> anyhow::Result<UploadSummary> {
        let mut summary = UploadSummary::default();
        let mut made_dirs: HashSet<String> = HashSet::new();
        self.mkdir_p(remote_root).await?;
        made_dirs.insert(remote_root.to_string());

        for entry in walkdir::WalkDir::new(local_root).sort_by_file_name() {
            if should_stop() {
                bail!("上传已取消（已传文件保留 .rpart 断点，可续传）");
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
                if !made_dirs.contains(&remote_path) {
                    self.mkdir_p(&remote_path).await?;
                    made_dirs.insert(remote_path);
                }
                continue;
            }
            let local_size = entry.metadata()?.len();
            if let Some(remote_size) = self.metadata_size(&remote_path).await {
                if remote_size == local_size {
                    summary.skipped += 1;
                    on_progress(local_size); // already present — count as done
                    tracing::info!("跳过（大小一致）: {remote_path}");
                    continue;
                }
            }
            self.upload_file(entry.path(), &remote_path, local_size, on_progress)
                .await?;
            summary.files += 1;
            summary.bytes += local_size;
            tracing::info!("完成: {remote_path} ({})", crate::model::human_bytes(local_size));
        }
        Ok(summary)
    }

    async fn upload_file(
        &self,
        local: &Path,
        remote_path: &str,
        local_size: u64,
        on_progress: &(dyn Fn(u64) + Send + Sync),
    ) -> anyhow::Result<()> {
        let part_path = format!("{remote_path}.rpart");
        let start = match self.metadata_size(&part_path).await {
            Some(part_size) if part_size < local_size => part_size,
            _ => 0,
        };

        let mut local_file = tokio::fs::File::open(local)
            .await
            .with_context(|| format!("打开本地文件 {}", local.display()))?;
        local_file.seek(std::io::SeekFrom::Start(start)).await?;

        let flags = if start > 0 {
            OpenFlags::CREATE | OpenFlags::WRITE
        } else {
            OpenFlags::CREATE | OpenFlags::WRITE | OpenFlags::TRUNCATE
        };
        let mut remote = self
            .sftp
            .open_with_flags(&part_path, flags)
            .await
            .with_context(|| format!("打开远端文件 {part_path}"))?;
        if start > 0 {
            remote.seek(std::io::SeekFrom::Start(start)).await?;
        }

        let mut buf = vec![0u8; CHUNK];
        let mut sent = start;
        loop {
            let n = local_file.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            remote.write_all(&buf[..n]).await?;
            sent += n as u64;
            on_progress(n as u64);
        }
        remote
            .shutdown()
            .await
            .context("关闭远端文件句柄失败")?;

        // rename may refuse to overwrite — clear the destination first
        let _ = self.sftp.remove_file(remote_path).await;
        self.sftp
            .rename(&part_path, remote_path)
            .await
            .with_context(|| format!("重命名 {part_path} -> {remote_path}"))?;
        let _ = sent;
        Ok(())
    }

    pub async fn remove_dir(&self, path: &str) -> anyhow::Result<()> {
        self.sftp
            .remove_dir(path)
            .await
            .with_context(|| format!("删除远端目录 {path} 失败"))
    }

    /// Recursive delete of a remote directory. Returns the number of files
    /// removed. Caller must ensure the path is inside the managed root.
    pub async fn remove_tree(&self, remote_dir: &str) -> anyhow::Result<u64> {
        let mut removed = 0u64;
        let entries: Vec<_> = match self.sftp.read_dir(remote_dir).await {
            Ok(rd) => rd.collect(),
            Err(_) => return Ok(0),
        };
        for entry in entries {
            let path = entry.path();
            let meta = entry.metadata();
            if meta.is_dir() {
                removed += Box::pin(self.remove_tree(&path)).await?;
                let _ = self.sftp.remove_dir(&path).await;
            } else {
                let _ = self.sftp.remove_file(&path).await;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Recursive size of a remote directory (bytes).
    pub async fn dir_size(&self, remote_dir: &str) -> anyhow::Result<u64> {
        let (code, out) = self.exec(&format!("du -sb {}", shell_quote(remote_dir))).await?;
        if code == 0 {
            if let Some(first) = out.lines().next() {
                if let Some(size) = first.split('\t').next().and_then(|s| s.parse().ok()) {
                    return Ok(size);
                }
            }
        }
        // fallback: pure SFTP walk
        let mut total = 0u64;
        let entries: Vec<_> = match self.sftp.read_dir(remote_dir).await {
            Ok(rd) => rd.collect(),
            Err(_) => return Ok(0),
        };
        for entry in entries {
            let meta = entry.metadata();
            if meta.is_dir() {
                total += Box::pin(self.dir_size(&entry.path())).await?;
            } else {
                total += meta.len();
            }
        }
        Ok(total)
    }

    /// Managed-root children: (name, size_bytes) sorted oldest (name asc) first.
    pub async fn managed_entries(&self, root: &str) -> anyhow::Result<Vec<(String, u64)>> {
        let mut out = Vec::new();
        let entries: Vec<_> = match self.sftp.read_dir(root).await {
            Ok(rd) => rd.collect(),
            Err(_) => return Ok(out),
        };
        for entry in entries {
            let name = entry.file_name();
            if name.starts_with('.') {
                continue;
            }
            let meta = entry.metadata();
            if meta.is_dir() {
                let size = self.dir_size(&entry.path()).await.unwrap_or(0);
                out.push((name, size));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }
}

fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_quote_escapes() {
        assert_eq!(shell_quote("/a b/c"), "'/a b/c'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }
}
