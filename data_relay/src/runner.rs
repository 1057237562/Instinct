//! Task execution and the download → boot 无卡 → upload → power-off pipeline.

use crate::autodl::AutoDl;
use crate::config::Config;
use crate::disk::{self, DiskUsage};
use crate::jupyter::JupyterUploader;
use crate::model::{self, Task, TaskStatus, UploadMode};
use crate::pull;
use crate::sftp::SftpUploader;
use crate::store::TaskStore;
use anyhow::{Context, bail};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Child;
use tokio::sync::Mutex;

struct AbortOnDrop(tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) { self.0.abort(); }
}

pub struct Relay {
    pub config_path: PathBuf,
    pub data_dir: PathBuf,
    pub cfg: std::sync::RwLock<Config>,
    pub store: Arc<TaskStore>,
    children: Mutex<HashMap<String, Child>>,
    cancelling: Mutex<HashSet<String>>,
    pub training_transition: Arc<Mutex<()>>,
    pub training_operations_lock: std::sync::Mutex<()>,
    pub data_locks: std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl Relay {
    pub fn new(config_path: PathBuf, cfg: Config) -> anyhow::Result<Arc<Self>> {
        let data_dir = std::path::absolute(&cfg.server.data_dir)?;
        let store = Arc::new(TaskStore::open(&data_dir.join("tasks"))?);
        let relay = Arc::new(Relay {
            config_path,
            data_dir,
            cfg: std::sync::RwLock::new(cfg),
            store,
            children: Mutex::new(HashMap::new()),
            cancelling: Mutex::new(HashSet::new()),
            training_transition: Arc::new(Mutex::new(())),
            training_operations_lock: std::sync::Mutex::new(()),
            data_locks: std::sync::Mutex::new(HashMap::new()),
        });
        Ok(relay)
    }

    pub fn config(&self) -> Config {
        self.cfg.read().unwrap().clone()
    }

    pub fn persist_config(&self, cfg: &Config) -> anyhow::Result<()> {
        cfg.validate()?;
        cfg.save(&self.config_path)?;
        *self.cfg.write().unwrap() = cfg.clone();
        Ok(())
    }

    // ---- task creation / lifecycle -------------------------------------------

    pub async fn create_task(
        self: &Arc<Self>,
        name: &str,
        script_name: &str,
        script_bytes: &[u8],
        output_dir: &str,
        auto_upload: bool,
        upload_mode: UploadMode,
        autostart: bool,
    ) -> anyhow::Result<Task> {
        let script_name = sanitize_filename(script_name)?;
        let ext = script_name
            .rsplit('.')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let cfg = self.config();
        let interpreter = match ext.as_str() {
            "py" => cfg.runner.python.clone(),
            "sh" => "bash".to_string(),
            other => bail!("仅支持 .py / .sh 脚本，收到 .{other}"),
        };

        let id = format!("t_{}_{:04x}", date_stamp(), rand_suffix());
        let dir = self.store.task_dir(&id);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join(&script_name), script_bytes)?;

        let task = Task {
            id: id.clone(),
            name: if name.trim().is_empty() {
                script_name.clone()
            } else {
                name.trim().to_string()
            },
            status: TaskStatus::Queued,
            created_at: model::now_secs(),
            started_at: None,
            finished_at: None,
            script: script_name,
            interpreter,
            output_dir: output_dir.trim().trim_matches('/').to_string(),
            exit_code: None,
            auto_upload,
            upload_mode,
            remote_path: None,
            upload_total_bytes: 0,
            upload_done_bytes: 0,
            upload_transferred_bytes: 0,
            upload_attempt: 0,
            upload_phase: String::new(),
            pull_token: None,
            error: None,
            data_upload: None,
        };
        self.store.insert(task.clone())?;
        if autostart {
            self.start_task(&id).await?;
        }
        Ok(self.store.get(&task.id).unwrap_or(task))
    }

    /// Run the download script of a task (any non-running state).
    pub async fn start_task(self: &Arc<Self>, id: &str) -> anyhow::Result<()> {
        let task = self
            .store
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("任务 {id} 不存在"))?;
        if matches!(task.status, TaskStatus::Running | TaskStatus::Uploading) {
            bail!("任务 {id} 正在运行");
        }
        let dir = self.store.task_dir(id);
        let script_path = dir.join(&task.script);
        if task.data_upload.is_some() {
            bail!("数据任务无需执行脚本，请使用继续交付");
        }
        if !script_path.is_file() {
            bail!("脚本 {} 不存在", script_path.display());
        }
        append_log(
            &dir,
            &format!(
                "[runner] {} 启动脚本 {} ({})\n",
                timestamp(),
                task.script,
                task.interpreter
            ),
        )?;
        self.store.update(id, |t| {
            t.status = TaskStatus::Running;
            t.started_at = Some(model::now_secs());
            t.finished_at = None;
            t.exit_code = None;
            t.error = None;
        })?;

        let this = Arc::clone(self);
        let id = id.to_string();
        tokio::spawn(async move {
            if let Err(e) = Arc::clone(&this).supervise(&id).await {
                append_log(&this.store.task_dir(&id), &format!("[runner] {e:#}\n")).ok();
                this.store.update(&id, |t| {
                    t.status = TaskStatus::Failed;
                    t.error = Some(format!("{e:#}"));
                    t.finished_at = Some(model::now_secs());
                })
                .ok();
            }
        });
        Ok(())
    }

    async fn supervise(self: Arc<Self>, id: &str) -> anyhow::Result<()> {
        let task = self.store.get(id).ok_or_else(|| anyhow::anyhow!("任务不存在"))?;
        let dir = self.store.task_dir(id);
        let cfg = self.config();

        let mut cmd = tokio::process::Command::new(&task.interpreter);
        cmd.arg(dir.join(&task.script))
            .current_dir(&dir)
            .env("RELAY_TASK_ID", id)
            .env("RELAY_TASK_DIR", &dir);
        for (k, v) in &cfg.runner.env {
            cmd.env(k, v);
        }
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "启动 {} 失败（解释器是否存在于部署服务器？）",
                task.interpreter
            )
        })?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        {
            let mut map = self.children.lock().await;
            map.insert(id.to_string(), child);
        }

        let log_path = dir.join("task.log");
        let log = Arc::new(Mutex::new(
            tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .await?,
        ));
        let mut writers = Vec::new();
        if let Some(out) = stdout {
            let log = Arc::clone(&log);
            writers.push(tokio::spawn(async move {
                let mut reader = tokio::io::BufReader::new(out);
                let mut buf = [0u8; 8192];
                loop {
                    match reader.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            log.lock().await.write_all(&buf[..n]).await.ok();
                        }
                    }
                }
            }));
        }
        if let Some(err_pipe) = stderr {
            let log = Arc::clone(&log);
            writers.push(tokio::spawn(async move {
                let mut reader = tokio::io::BufReader::new(err_pipe);
                let mut buf = [0u8; 8192];
                loop {
                    match reader.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            log.lock().await.write_all(&buf[..n]).await.ok();
                        }
                    }
                }
            }));
        }

        let exit_code;
        loop {
            {
                let mut map = self.children.lock().await;
                if let Some(c) = map.get_mut(id) {
                    if let Some(code) = c.try_wait()? {
                        exit_code = code.code().unwrap_or(-1);
                        map.remove(id);
                        break;
                    }
                } else {
                    bail!("子进程记录丢失");
                }
            }
            if self.cancelling.lock().await.contains(id) {
                let mut map = self.children.lock().await;
                if let Some(c) = map.get_mut(id) {
                    let _ = c.kill().await;
                }
                map.remove(id);
                self.cancelling.lock().await.remove(id);
                self.store.update_status(id, TaskStatus::Cancelled)?;
                append_log(&dir, &format!("[runner] {} 已取消\n", timestamp()))?;
                for w in writers {
                    let _ = w.await;
                }
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
        for w in writers {
            let _ = w.await;
        }

        self.store.update(id, |t| t.exit_code = Some(exit_code))?;
        append_log(&dir, &format!("[runner] {} 脚本退出码 {exit_code}\n", timestamp()))?;

        if exit_code != 0 {
            self.store.update(id, |t| {
                t.status = TaskStatus::Failed;
                t.error = Some(format!("脚本退出码 {exit_code}"));
                t.finished_at = Some(model::now_secs());
            })?;
            return Ok(());
        }

        self.store.update_status(id, TaskStatus::DownloadDone)?;
        if self.store.get(id).map(|t| t.auto_upload).unwrap_or(false) {
            if let Err(e) = Arc::clone(&self).run_pipeline(id).await {
                append_log(&dir, &format!("[pipeline] {e:#}\n"))?;
                self.store.update(id, |t| {
                    t.status = TaskStatus::Failed;
                    t.error = Some(format!("{e:#}"));
                    t.finished_at = Some(model::now_secs());
                })?;
            }
        }
        Ok(())
    }

    pub async fn cancel(&self, id: &str) -> anyhow::Result<()> {
        let task = self
            .store
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("任务 {id} 不存在"))?;
        match task.status {
            TaskStatus::Running => {
                self.cancelling.lock().await.insert(id.to_string());
                let mut map = self.children.lock().await;
                if let Some(c) = map.get_mut(id) {
                    let _ = c.start_kill();
                }
            }
            TaskStatus::Uploading => {
                self.cancelling.lock().await.insert(id.to_string());
            }
            _ => bail!("任务 {id} 未在运行"),
        }
        Ok(())
    }

    // ---- upload pipeline ------------------------------------------------------

    /// Download finished → boot 无卡 → quota pre-check → upload → power off.
    pub async fn run_pipeline(self: Arc<Self>, id: &str) -> anyhow::Result<()> {
        let task = self
            .store
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("任务 {id} 不存在"))?;
        if task.status == TaskStatus::Uploading {
            bail!("任务 {id} 正在上传中");
        }
        // Serialize data delivery with GPU mode changes and shutdown. A new
        // data job waits safely while another job owns the transition guard.
        let _data_transition = if task.data_upload.is_some() {
            Some(Arc::clone(&self.training_transition).lock_owned().await)
        } else { None };
        let dir = self.store.task_dir(id);
        let mut cfg = self.config();
        if let Some(data) = &task.data_upload {
            cfg.autodl.instance_uuid = data.instance_uuid.clone();
            // Direct delivery leaves the selected instance available for training.
            cfg.autodl.power_off_after_upload = false;
        }
        let log = |line: String| {
            append_log(&dir, &line).ok();
        };
        self.cancelling.lock().await.remove(id);
        let stop = Arc::new(AtomicBool::new(false));
        let _stop_watcher;
        {
            let relay_for_stop = Arc::clone(&self);
            let id_for_stop = id.to_string();
            let stop_flag = Arc::clone(&stop);
            let watcher = tokio::spawn(async move {
                // map cancelling flag → stop flag for the upload loop
                loop {
                    if relay_for_stop
                        .cancelling
                        .lock()
                        .await
                        .contains(&id_for_stop)
                    {
                        stop_flag.store(true, Ordering::SeqCst);
                        return;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            });
            // Aborting the watcher on every return avoids a leaked task per upload.
            _stop_watcher = AbortOnDrop(watcher);
        }

        self.store.update(id, |t| {
            t.status = TaskStatus::Uploading;
            t.upload_done_bytes = 0;
            t.upload_transferred_bytes = 0;
            t.upload_attempt += 1;
            t.upload_phase = "preparing".into();
            t.error = None;
        })?;
        log(format!("[pipeline] {} 开始上传流程\n", timestamp()));

        let client = AutoDl::new(&cfg.autodl)?;
        log("[pipeline] 检查/启动 AutoDL 实例（无卡模式）…\n".into());
        let info = client.ensure_running().await?;
        log(format!(
            "[pipeline] 实例 {} 状态 {} (start_mode={:?})\n",
            info.uuid, info.status, info.start_mode
        ));

        let remote_root = if task.data_upload.is_some() {
            crate::data_upload::DATASET_ROOT.to_string()
        } else { cfg.autodl.remote_root.clone() };
        let remote_task = if task.data_upload.is_some() {
            crate::data_upload::remote_stage(id)
        } else { model::join_remote(&remote_root, id)
            .ok_or_else(|| anyhow::anyhow!("remote_root 配置非法"))? };

        // local payload
        let out_dir = if task.output_dir.is_empty() {
            dir.clone()
        } else {
            model::resolve_inside(&dir, &task.output_dir)
                .ok_or_else(|| anyhow::anyhow!("output_dir 越界: {}", task.output_dir))?
        };
        if !out_dir.is_dir() {
            bail!(
                "输出目录 {} 不存在：脚本应把产物写到该目录（可用 output_dir 字段调整）",
                out_dir.display()
            );
        }
        // 脚本可写 upload_paths.txt（相对任务目录，一行一个，# 为注释）精确声明
        // 清洗后的交付文件；未提供则上传整个 output_dir。声明的文件经硬链接
        // （跨设备时复制）汇集到 _relay_stage 暂存目录再上传。
        let manifest_path = dir.join("upload_paths.txt");
        let upload_root: PathBuf = if manifest_path.is_file() {
            let text = std::fs::read_to_string(&manifest_path)?;
            let mut files: Vec<PathBuf> = Vec::new();
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let p = model::resolve_inside(&dir, line)
                    .ok_or_else(|| anyhow::anyhow!("upload_paths.txt 路径越界: {line}"))?;
                if !p.is_file() {
                    bail!("upload_paths.txt 引用的文件不存在: {line}");
                }
                files.push(p);
            }
            if files.is_empty() {
                bail!("upload_paths.txt 存在但没有列出任何文件");
            }
            let stage = dir.join("_relay_stage");
            let _ = std::fs::remove_dir_all(&stage);
            std::fs::create_dir_all(&stage)?;
            for src in &files {
                let rel = src.strip_prefix(&dir)?;
                let dst = stage.join(rel);
                std::fs::create_dir_all(dst.parent().unwrap_or(&stage))?;
                if std::fs::hard_link(src, &dst).is_err() {
                    std::fs::copy(src, &dst)?;
                }
            }
            log(format!(
                "[pipeline] 脚本声明了 {} 个清洗产物，经暂存目录上传
",
                files.len()
            ));
            stage
        } else {
            out_dir.clone()
        };
        let local_bytes = model::dir_size(&upload_root);
        self.store.update(id, |t| t.upload_total_bytes = local_bytes)?;
        log(format!(
            "[pipeline] 本地待传 {} （{}）\n",
            upload_root.display(),
            model::human_bytes(local_bytes)
        ));

        // upload channel
        let ssh = info.ssh.clone().ok_or_else(|| {
            anyhow::anyhow!("实例信息缺少 SSH 凭据（proxy_host/ssh_port/root_password）")
        })?;
        let password = if cfg.autodl.ssh_password.is_empty() {
            ssh.password.clone()
        } else {
            cfg.autodl.ssh_password.clone()
        };
        let known_hosts_path = self.data_dir.join("known_hosts.json");

        match task.upload_mode {
            UploadMode::Pull => {
                let base = cfg.autodl.pull_base_url.trim_end_matches('/');
                if base.is_empty() {
                    bail!(
                        "pull 高速通道需要设置 autodl.pull_base_url（本机对外可达地址，\
                         如 http://<服务器IP>:7800），实例将用它多连接拉取数据"
                    );
                }
                let sftp = self
                    .connect_and_precheck(
                        &info, &ssh, &password, &known_hosts_path, &remote_root, local_bytes,
                        task.data_upload.as_ref().map(|_| remote_task.as_str()), &log,
                    )
                    .await?;
                crate::data_upload::remote_check(&sftp, &task, false).await?;

                // 1. scan local output and build the control-plane artifacts
                let files = pull::scan_local(&upload_root)?;
                if files.is_empty() {
                    bail!("输出目录 {} 没有可上传文件", out_dir.display());
                }
                let total = pull::manifest_total(&files);
                let token = pull::new_pull_token();
                let manifest_tsv = pull::build_manifest_tsv(&files);
                let aria2_input =
                    pull::build_aria2_input(base, id, &token, &remote_task, &files);
                let script = pull::build_pull_script(base, id, &token, &remote_task);
                let control_bytes =
                    (manifest_tsv.len() + aria2_input.len() + script.len()) as u64;
                std::fs::write(dir.join(pull::MANIFEST_TSV), &manifest_tsv)?;
                self.store.update(id, |t| {
                    t.pull_token = Some(token.clone());
                    t.upload_total_bytes = total;
                })?;

                // 2. push control files, launch the instance-side puller
                sftp
                    .write_file(
                        &format!("{remote_task}/{}", pull::MANIFEST_TSV),
                        manifest_tsv.as_bytes(),
                    )
                    .await?;
                sftp
                    .write_file(
                        &format!("{remote_task}/{}", pull::MANIFEST_ARIA2),
                        aria2_input.as_bytes(),
                    )
                    .await?;
                sftp
                    .write_file(
                        &format!("{remote_task}/{}", pull::PULL_SCRIPT),
                        script.as_bytes(),
                    )
                    .await?;
                // Allocated blocks, rather than apparent file lengths: aria2
                // can extend sparse files before their contents have arrived.
                let (_, initial) = sftp.exec(&format!("du -sB1 '{remote_task}' 2>/dev/null | cut -f1")).await?;
                let mut last_done = initial.trim().parse::<u64>().unwrap_or(0).saturating_sub(control_bytes).min(total);
                self.store.update(id, |t| {
                    t.upload_done_bytes = last_done;
                    t.upload_phase = "transferring".into();
                })?;
                let (_, launch_out) = sftp
                    .exec(&format!(
                        "chmod +x '{remote_task}/{}' && nohup setsid bash '{remote_task}/{}' >/dev/null 2>&1 & echo RELAY-LAUNCHED",
                        pull::PULL_SCRIPT,
                        pull::PULL_SCRIPT
                    ))
                    .await?;
                if !launch_out.contains("RELAY-LAUNCHED") {
                    bail!("实例端拉取进程启动失败: {}", launch_out.trim());
                }
                log(format!(
                    "[pipeline] 高速通道已启动：{} 个文件 / {}，实例从 {base} 多连接拉取\n",
                    files.len(),
                    model::human_bytes(total)
                ));

                // 3. poll progress until the instance reports completion
                let mut stall_ticks = 0u32;
                loop {
                    if stop.load(Ordering::SeqCst) {
                        let _ = sftp
                            .exec(&format!(
                                "if test -f '{remote_task}/_relay_pull.pid'; then pid=$(cat '{remote_task}/_relay_pull.pid'); case \"$pid\" in ''|*[!0-9]*) exit 1;; esac; kill -- -\"$pid\" 2>/dev/null || true; fi"
                            ))
                            .await;
                        bail!("上传已取消（已拉取部分保留，可断点续传）");
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    let (_, probe) = sftp
                        .exec(&format!(
                            "du -sB1 '{remote_task}' 2>/dev/null | cut -f1; grep -o 'RELAY-PULL-EXIT=[0-9]*' '{remote_task}/{}' 2>/dev/null | tail -1",
                            pull::PULL_LOG
                        ))
                        .await?;
                    let mut lines = probe.lines();
                    let used: u64 = lines
                        .next()
                        .and_then(|l| l.trim().parse().ok())
                        .unwrap_or(0);
                    let done = used.saturating_sub(control_bytes).min(total);
                    if done != last_done {
                        self.store.update(id, |t| {
                            t.upload_done_bytes = done;
                            t.upload_transferred_bytes += done.saturating_sub(last_done);
                        }).ok();
                        last_done = done;
                        stall_ticks = 0;
                    } else {
                        stall_ticks += 1;
                    }
                    if let Some(marker) = lines.next() {
                        let rc: i32 = marker
                            .trim()
                            .trim_start_matches("RELAY-PULL-EXIT=")
                            .parse()
                            .unwrap_or(1);
                        if rc == 0 {
                            self.store.update(id, |t| {
                                t.upload_done_bytes = total;
                                t.upload_phase = "verifying".into();
                            }).ok();
                            log("[pipeline] 实例拉取完成，逐文件大小校验通过 ✓\n".into());
                        } else {
                            let (_, tail) = sftp
                                .exec(&format!(
                                    "tail -c 2000 '{remote_task}/{}'",
                                    pull::PULL_LOG
                                ))
                                .await?;
                            bail!("实例拉取失败 (rc={rc})，远端日志尾部:\n{}", tail.trim());
                        }
                        break;
                    }
                    if stall_ticks >= 360 {
                        bail!("拉取进度约 30 分钟无增长，疑似通道故障；断点已保留，可重试");
                    }
                }
                crate::data_upload::remote_check(&sftp, &task, true).await?;
            }
            UploadMode::Sftp => {
                let sftp = self
                    .connect_and_precheck(
                        &info, &ssh, &password, &known_hosts_path, &remote_root, local_bytes,
                        task.data_upload.as_ref().map(|_| remote_task.as_str()), &log,
                    )
                    .await?;
                crate::data_upload::remote_check(&sftp, &task, false).await?;
                self.store.update(id, |t| t.upload_phase = "transferring".into())?;
                let store = Arc::clone(&self.store);
                let id_owned = id.to_string();
                let progress = move |delta: u64, resumed: bool| {
                    store.update(&id_owned, |t| {
                        t.upload_done_bytes += delta;
                        if !resumed { t.upload_transferred_bytes += delta; }
                    }).ok();
                };
                let stop_flag = Arc::clone(&stop);
                let should_stop = move || stop_flag.load(Ordering::SeqCst);
                let summary = sftp
                    .upload_tree(&upload_root, &remote_task, &progress, &should_stop)
                    .await?;
                self.store.update(id, |t| t.upload_phase = "verifying".into())?;
                log(format!(
                    "[pipeline] 上传完成：新传 {} 个文件 / 跳过 {} 个 / 本轮 {} \n",
                    summary.files,
                    summary.skipped,
                    model::human_bytes(summary.bytes)
                ));
                crate::data_upload::remote_check(&sftp, &task, true).await?;
            }
            UploadMode::Jupyter => {
                let jupyter = info
                    .jupyter
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("实例信息缺少 Jupyter 地址/token"))?;
                let up = JupyterUploader::new(&jupyter);
                up.check().await?;
                let max_file = cfg.disk.jupyter_max_file_mb * 1024 * 1024;
                let usage = DiskUsage {
                    total: info.data_disk_total.unwrap_or(0),
                    used: info.data_disk_used.unwrap_or(0),
                    source: "api/snapshot",
                };
                let decision = disk::plan(
                    &usage,
                    local_bytes,
                    cfg.disk.max_used_ratio,
                    cfg.disk.reserved_gb,
                );
                if !decision.allowed {
                    self.store.update(id, |t| t.error = Some(decision.message.clone()))?;
                    bail!("{}", decision.message);
                }
                let store = Arc::clone(&self.store);
                let id_owned = id.to_string();
                self.store.update(id, |t| t.upload_phase = "transferring".into())?;
                let progress = move |delta: u64| {
                    store.update(&id_owned, |t| {
                        t.upload_done_bytes += delta;
                        t.upload_transferred_bytes += delta;
                    }).ok();
                };
                let stop_flag = Arc::clone(&stop);
                let should_stop = move || stop_flag.load(Ordering::SeqCst);
                let summary = up
                    .upload_tree(&upload_root, &remote_task, max_file, &progress, &should_stop)
                    .await?;
                log(format!("[pipeline] Jupyter 上传完成：{} 个文件\n", summary.files));
            }
        }

        self.store.update(id, |t| {
            t.upload_total_bytes = local_bytes;
            t.upload_done_bytes = local_bytes;
            t.remote_path = Some(if task.data_upload.is_some() {
                crate::data_upload::DATASET_ROOT.to_string()
            } else { remote_task.clone() });
            t.status = TaskStatus::Uploaded;
        })?;

        if cfg.autodl.power_off_after_upload {
            let _shutdown_guard = match crate::training::guard_shutdown(&self, &info.uuid).await {
                Ok(guard) => guard,
                Err(error) => {
                    log(format!("[pipeline] 保持实例运行，关机检查未通过: {error:#}\n"));
                    self.store.update_status(id, TaskStatus::Done)?;
                    return Ok(());
                }
            };
            log("[pipeline] 上传完成，关闭实例（省无卡机时费）…\n".into());
            match client.power_off(&info.uuid).await {
                Ok(()) => log("[pipeline] 已发送关机指令\n".into()),
                Err(e) => log(format!("[pipeline] 关机失败（忽略）: {e:#}\n")),
            }
        }
        self.store.update_status(id, TaskStatus::Done)?;
        log(format!("[pipeline] {} 任务完成 ✓\n", timestamp()));
        Ok(())
    }

    /// Delete oldest managed dirs (name asc = oldest first) until
    /// `used - freed + incoming` fits below the quota limit. Never deletes the
    /// newest `keep_last_uploads` entries.
    async fn cleanup_managed(
        &self,
        sftp: &SftpUploader,
        remote_root: &str,
        usage: &DiskUsage,
        incoming: u64,
    ) -> anyhow::Result<u64> {
        let cfg = self.config();
        let limit = ((usage.total as f64 * cfg.disk.max_used_ratio) as u64)
            .saturating_sub((cfg.disk.reserved_gb * 1024.0 * 1024.0 * 1024.0) as u64);
        let entries = sftp.managed_entries(remote_root).await?;
        let mut freed = 0u64;
        let mut remaining = entries.len() as u32;
        for (name, size) in entries {
            if remaining <= cfg.disk.keep_last_uploads {
                break;
            }
            if disk::fits_after_free(usage, freed, incoming, limit) {
                break;
            }
            let path = model::join_remote(remote_root, &name)
                .ok_or_else(|| anyhow::anyhow!("非法托管路径 {name}"))?;
            tracing::info!("清理托管目录 {path}（{}）", model::human_bytes(size));
            sftp.remove_tree(&path).await?;
            let _ = sftp.remove_dir(&path).await;
            freed += size;
            remaining -= 1;
        }
        if freed > 0 {
            tracing::info!("共释放 {}", model::human_bytes(freed));
        }
        Ok(freed)
    }

    /// SSH connect (with host-key bookkeeping) + managed-root quota pre-check.
    /// Shared by the pull and sftp channels — for pull this is control plane
    /// only; bulk data never traverses this connection.
    #[allow(clippy::too_many_arguments)]
    async fn connect_and_precheck<F: Fn(String)>(
        &self,
        info: &crate::autodl::InstanceInfo,
        ssh: &crate::autodl::SshInfo,
        password: &str,
        known_hosts_path: &Path,
        remote_root: &str,
        incoming_bytes: u64,
        existing_stage: Option<&str>,
        log: &F,
    ) -> anyhow::Result<SftpUploader> {
        let cfg = self.config();
        let (sftp, fingerprint) = SftpUploader::connect(
            &ssh.host,
            ssh.port,
            &cfg.autodl.ssh_user,
            password,
            cfg.autodl.strict_host_key,
            read_known_host(known_hosts_path, &ssh.host, ssh.port),
        )
        .await?;
        if !fingerprint.is_empty() {
            write_known_host(known_hosts_path, &ssh.host, ssh.port, &fingerprint);
            log(format!("[pipeline] SSH 主机指纹 {fingerprint}\n"));
        }
        sftp.mkdir_p(remote_root).await?;
        let usage = match sftp.fs_usage(remote_root).await {
            Ok((total, free)) => DiskUsage {
                total,
                used: total.saturating_sub(free),
                source: "sftp/fs_info",
            },
            Err(_) => DiskUsage {
                total: info.data_disk_total.unwrap_or(0),
                used: info.data_disk_used.unwrap_or(0),
                source: "api/snapshot",
            },
        };
        log(format!(
            "[pipeline] 数据盘 ({}) 总量 {} 已用 {} 剩余 {}\n",
            usage.source,
            model::human_bytes(usage.total),
            model::human_bytes(usage.used),
            model::human_bytes(usage.free())
        ));
        // Account for already allocated staging blocks on a resumed upload.
        // du (without --apparent-size) also handles sparse aria2 files correctly.
        let staged_bytes = if let Some(stage) = existing_stage {
            let (code, output) = sftp.exec(&format!("du -sB1 '{stage}' 2>/dev/null")).await?;
            if code == 0 { output.split_whitespace().next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) } else { 0 }
        } else { 0 };
        let incoming_bytes = incoming_bytes.saturating_sub(staged_bytes);
        let mut decision = disk::plan(
            &usage,
            incoming_bytes,
            cfg.disk.max_used_ratio,
            cfg.disk.reserved_gb,
        );
        if !decision.allowed && cfg.disk.auto_clean && remote_root != crate::data_upload::DATASET_ROOT {
            log("[pipeline] 超出限额，尝试清理托管目录…\n".into());
            let freed = self
                .cleanup_managed(&sftp, remote_root, &usage, incoming_bytes)
                .await?;
            let usage2 = DiskUsage {
                used: usage.used.saturating_sub(freed),
                ..usage
            };
            decision = disk::plan(
                &usage2,
                incoming_bytes,
                cfg.disk.max_used_ratio,
                cfg.disk.reserved_gb,
            );
        }
        if !decision.allowed {
            bail!("{}", decision.message);
        }
        log(format!("[pipeline] 配额预检通过：{}\n", decision.message));
        Ok(sftp)
    }

    /// Local directory served over the pull channel for a task.
    pub fn pull_root(&self, id: &str) -> Option<PathBuf> {
        let task = self.store.get(id)?;
        let dir = self.store.task_dir(id);
        if dir.join("upload_paths.txt").is_file() && dir.join("_relay_stage").is_dir() {
            return Some(dir.join("_relay_stage"));
        }
        if task.output_dir.is_empty() {
            Some(dir)
        } else {
            model::resolve_inside(&dir, &task.output_dir)
        }
    }

    // ---- ad-hoc AutoDL helpers for the WebUI -----------------------------------

    /// 账号下实例列表（脱敏），供 WebUI 下拉选择；selected 为当前配置选中的 uuid。
    pub async fn autodl_instances(&self) -> anyhow::Result<serde_json::Value> {
        let cfg = self.config();
        let client = AutoDl::new(&cfg.autodl)?;
        let all = client.list().await?;
        let selected = cfg.autodl.instance_uuid.clone();
        let items: Vec<serde_json::Value> = all
            .iter()
            .map(|i| {
                json!({
                    "uuid": i.uuid,
                    "name": i.name,
                    "status": i.status,
                    "start_mode": i.start_mode,
                    "charge_type": i.charge_type,
                    "selected": !selected.is_empty() && selected == i.uuid,
                })
            })
            .collect();
        Ok(json!({"selected": selected, "count": items.len(), "instances": items}))
    }

    pub async fn autodl_status(&self) -> anyhow::Result<serde_json::Value> {
        let cfg = self.config();
        let client = AutoDl::new(&cfg.autodl)?;
        let info = client.pick_instance().await?;
        Ok(info.to_redacted_json())
    }

    pub async fn autodl_boot_no_gpu(&self) -> anyhow::Result<serde_json::Value> {
        let cfg = self.config();
        let client = AutoDl::new(&cfg.autodl)?;
        let info = client.pick_instance().await?;
        if info.is_running() {
            return Ok(json!({"ok": true, "message": "实例已在运行", "status": info.status}));
        }
        client.power_on_no_gpu(&info.uuid).await?;
        Ok(json!({"ok": true, "message": "已发送无卡模式开机指令", "status": info.status}))
    }

    pub async fn autodl_power_off(&self) -> anyhow::Result<serde_json::Value> {
        let cfg = self.config();
        let client = AutoDl::new(&cfg.autodl)?;
        let info = client.pick_instance().await?;
        let _shutdown_guard = crate::training::guard_shutdown(self, &info.uuid).await?;
        client.power_off(&info.uuid).await?;
        Ok(json!({"ok": true, "message": "已发送关机指令"}))
    }

    /// Disk report: API numbers when the instance is off, live SFTP numbers
    /// (plus managed-dir breakdown) when it is running.
    pub async fn autodl_disk(&self) -> anyhow::Result<serde_json::Value> {
        let cfg = self.config();
        let client = AutoDl::new(&cfg.autodl)?;
        let info = client.pick_instance().await?;
        let mut report = json!({
            "instance": info.uuid,
            "status": info.status,
            "data_disk_total": info.data_disk_total,
            "data_disk_used": info.data_disk_used,
            "remote_root": cfg.autodl.remote_root,
        });
        if info.is_running() {
            if let Some(ssh) = &info.ssh {
                let password = if cfg.autodl.ssh_password.is_empty() {
                    ssh.password.clone()
                } else {
                    cfg.autodl.ssh_password.clone()
                };
                if let Ok((sftp, _)) = SftpUploader::connect(
                    &ssh.host,
                    ssh.port,
                    &cfg.autodl.ssh_user,
                    &password,
                    cfg.autodl.strict_host_key,
                    read_known_host(&self.data_dir.join("known_hosts.json"), &ssh.host, ssh.port),
                )
                .await
                {
                    let root = &cfg.autodl.remote_root;
                    if let Ok((total, free)) = sftp.fs_usage(root).await {
                        report["disk"] = json!({
                            "total": total, "used": total.saturating_sub(free), "free": free,
                            "source": "sftp/fs_info",
                        });
                    }
                    if let Ok(entries) = sftp.managed_entries(root).await {
                        report["managed"] = json!(entries.iter()
                            .map(|(n, s)| json!({"name": n, "bytes": s}))
                            .collect::<Vec<_>>());
                    }
                }
            }
        }
        Ok(report)
    }

    /// Delete a managed remote path (must be inside remote_root, not the root
    /// itself). Only used by the WebUI trash buttons.
    pub async fn delete_remote(&self, rel: &str) -> anyhow::Result<()> {
        let cfg = self.config();
        let root = &cfg.autodl.remote_root;
        let path = model::join_remote(root, rel)
            .filter(|p| p != root)
            .ok_or_else(|| anyhow::anyhow!("路径越界或指向托管根目录"))?;
        let client = AutoDl::new(&cfg.autodl)?;
        let info = client.pick_instance().await?;
        if !info.is_running() {
            bail!("实例未开机，无法清理");
        }
        let ssh = info.ssh.clone().ok_or_else(|| anyhow::anyhow!("缺少 SSH 凭据"))?;
        let password = if cfg.autodl.ssh_password.is_empty() {
            ssh.password.clone()
        } else {
            cfg.autodl.ssh_password.clone()
        };
        let (sftp, _) = SftpUploader::connect(
            &ssh.host,
            ssh.port,
            &cfg.autodl.ssh_user,
            &password,
            cfg.autodl.strict_host_key,
            read_known_host(&self.data_dir.join("known_hosts.json"), &ssh.host, ssh.port),
        )
        .await?;
        let removed = sftp.remove_tree(&path).await?;
        let _ = sftp.remove_dir(&path).await;
        tracing::info!("手动清理 {path}（{removed} 个文件）");
        Ok(())
    }

    pub async fn list_remote(&self) -> anyhow::Result<serde_json::Value> {
        let cfg = self.config();
        let client = AutoDl::new(&cfg.autodl)?;
        let info = client.pick_instance().await?;
        if !info.is_running() {
            bail!("实例未开机（无卡模式开机后可查看）");
        }
        let ssh = info.ssh.clone().ok_or_else(|| anyhow::anyhow!("缺少 SSH 凭据"))?;
        let password = if cfg.autodl.ssh_password.is_empty() {
            ssh.password.clone()
        } else {
            cfg.autodl.ssh_password.clone()
        };
        let (sftp, _) = SftpUploader::connect(
            &ssh.host,
            ssh.port,
            &cfg.autodl.ssh_user,
            &password,
            cfg.autodl.strict_host_key,
            read_known_host(&self.data_dir.join("known_hosts.json"), &ssh.host, ssh.port),
        )
        .await?;
        let entries = sftp
            .managed_entries(&cfg.autodl.remote_root)
            .await?
            .into_iter()
            .map(|(name, bytes)| json!({"name": name, "bytes": bytes}))
            .collect::<Vec<_>>();
        Ok(json!({"remote_root": cfg.autodl.remote_root, "entries": entries}))
    }
}

// ---- small helpers -------------------------------------------------------------

fn sanitize_filename(name: &str) -> anyhow::Result<String> {
    let name = name.replace('\\', "/");
    let name = name.rsplit('/').next().unwrap_or_default();
    if name.is_empty() || name.starts_with('.') || !name.contains('.') {
        bail!("脚本文件名非法: {name:?}");
    }
    Ok(name.to_string())
}

fn append_log(dir: &Path, line: &str) -> anyhow::Result<()> {
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("task.log"))?;
    f.write_all(line.as_bytes())?;
    Ok(())
}

fn timestamp() -> String {
    let secs = model::now_secs();
    let (y, mo, d, h, mi, s) = split_epoch(secs);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}")
}

/// `yyyymmdd_hhmmss` in UTC — used in task ids so name-sort equals age-sort.
fn date_stamp() -> String {
    let secs = model::now_secs();
    let (y, mo, d, h, mi, s) = split_epoch(secs);
    format!("{y:04}{mo:02}{d:02}_{h:02}{mi:02}{s:02}")
}

fn rand_suffix() -> u32 {
    // cheap per-call entropy; ids only need local uniqueness
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    nanos & 0xffff
}

/// Days-since-epoch civil calendar conversion (Howard Hinnant's algorithm).
fn split_epoch(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86400);
    let secs_of_day = secs.rem_euclid(86400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    let h = (secs_of_day / 3600) as u32;
    let mi = ((secs_of_day % 3600) / 60) as u32;
    let s = (secs_of_day % 60) as u32;
    (y, m, d, h, mi, s)
}

fn read_known_host(path: &Path, host: &str, port: u16) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let map: HashMap<String, String> = serde_json::from_str(&text).ok()?;
    map.get(&format!("{host}:{port}")).cloned()
}

fn write_known_host(path: &Path, host: &str, port: u16, fingerprint: &str) {
    let mut map: HashMap<String, String> = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    map.insert(format!("{host}:{port}"), fingerprint.to_string());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, serde_json::to_vec_pretty(&map).unwrap_or_default()).ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_filenames() {
        assert_eq!(sanitize_filename("run.py").unwrap(), "run.py");
        assert_eq!(sanitize_filename("a/b/run.sh").unwrap(), "run.sh");
        // basename extraction strips any path components — no traversal possible
        assert_eq!(sanitize_filename("..\\evil.py").unwrap(), "evil.py");
        assert!(sanitize_filename(".hidden").is_err());
        assert!(sanitize_filename("noext").is_err());
        assert!(sanitize_filename("").is_err());
    }

    #[test]
    fn epoch_split_known_values() {
        // 2026-10-05 00:00:00 UTC == 1791158400
        let (y, m, d, h, mi, s) = split_epoch(1_791_158_400);
        assert_eq!((y, m, d, h, mi, s), (2026, 10, 5, 0, 0, 0));
        let (_, _, _, h, mi, s) = split_epoch(1_791_158_400 + 16 * 3600 + 14 * 60 + 9);
        assert_eq!((h, mi, s), (16, 14, 9));
    }

    #[test]
    fn known_host_roundtrip() {
        let path = std::env::temp_dir().join(format!("kh_{}", uuid::Uuid::new_v4()));
        assert!(read_known_host(&path, "h", 1).is_none());
        write_known_host(&path, "h", 1, "SHA256:abc");
        assert_eq!(read_known_host(&path, "h", 1).as_deref(), Some("SHA256:abc"));
        std::fs::remove_file(&path).ok();
    }
}
