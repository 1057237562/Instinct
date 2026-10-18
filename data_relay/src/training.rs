//! SSH-backed training manager; the remote supervisor owns the process lifetime.
use crate::{autodl::AutoDl, config::Config, runner::Relay, sftp::SftpUploader};
use anyhow::{bail, Context};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

const HELPER: &str = include_str!("training_remote.py");

#[derive(Debug)]
struct QueueCancelled;
impl std::fmt::Display for QueueCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("训练排队已取消") }
}
impl std::error::Error for QueueCancelled {}

fn launch_defaults(mut input: Value) -> Value {
    // A stale Config WebUI checkbox must never silently disable the relay policy.
    input["shutdown_after_train"] = json!(true);
    input["shutdown_delay_seconds"] = json!(60);
    input
}

fn check_cancelled(path: Option<&PathBuf>) -> anyhow::Result<()> {
    if path.is_some_and(|path| path.with_extension("cancel").exists()) { return Err(QueueCancelled.into()); }
    Ok(())
}

fn terminal(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "interrupted" | "cancelled")
}

fn validate_start(input: &Value) -> anyhow::Result<()> {
    let path = input["args"]["data_path"]
        .as_str()
        .context("请选择训练语料")?;
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    if ![".jsonl", ".jsonl.gz", ".parquet"]
        .iter()
        .any(|suffix| name.ends_with(suffix))
    {
        bail!("语料必须是 JSONL、gzip JSONL 或 Parquet 文件");
    }
    Ok(())
}

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

async fn remote(sftp: &SftpUploader, payload: &Value) -> anyhow::Result<Value> {
    let command = format!("export PATH=\"/root/miniconda3/bin:/root/miniconda/bin:/opt/conda/bin:$PATH\"; python3 -c {} {}",
                          quote(HELPER), quote(&payload.to_string()));
    let (code, output) = sftp.exec(&command).await?;
    if code != 0 {
        bail!("服务器训练管理失败：{}", output.trim());
    }
    serde_json::from_str(output.trim()).context("训练管理器响应不是有效 JSON")
}

pub(crate) async fn connect(
    relay: &Relay,
    cfg: &Config,
    info: &crate::autodl::InstanceInfo,
) -> anyhow::Result<SftpUploader> {
    let ssh = info.ssh.as_ref().context("实例没有 SSH 凭据")?;
    let password = if cfg.autodl.ssh_password.is_empty() {
        &ssh.password
    } else {
        &cfg.autodl.ssh_password
    };
    let known_path = relay.data_dir.join("known_hosts.json");
    let mut known: HashMap<String, String> = if known_path.exists() {
        serde_json::from_slice(&std::fs::read(&known_path)?)?
    } else {
        HashMap::new()
    };
    let key = format!("{}:{}", ssh.host, ssh.port);
    let (sftp, fingerprint) = SftpUploader::connect(
        &ssh.host,
        ssh.port,
        &cfg.autodl.ssh_user,
        password,
        true,
        known.get(&key).cloned(),
    )
    .await?;
    known.insert(key, fingerprint);
    let known_temp = known_path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
    std::fs::write(&known_temp, serde_json::to_vec_pretty(&known)?)?;
    std::fs::rename(known_temp, known_path)?;
    Ok(sftp)
}

fn payload(cfg: &Config, instance: &str, action: &str) -> Value {
    json!({"repo_root":cfg.training.repo_root, "python":cfg.training.python,
           "min_free_gb":cfg.training.min_free_gb, "cache_max_gb":cfg.training.cache_max_gb,
           "gpu_devices":"", "instance_uuid":instance, "action":action})
}

pub(crate) async fn assert_idle(connection: &SftpUploader, cfg: &Config, instance: &str) -> anyhow::Result<()> {
    remote(connection, &payload(cfg, instance, "assert_idle")).await?;
    Ok(())
}

pub async fn request(relay: &Relay, action: &str, input: Value) -> anyhow::Result<Value> {
    let cfg = relay.config();
    let mut input = if action == "start" {
        crate::training_cache::apply(&cfg, &relay.config_path,
            crate::training_panel::apply(&cfg, &relay.config_path, input)?)?
    } else {
        input
    };
    if matches!(action, "start" | "resume") { input = launch_defaults(input); }
    if action == "start" {
        validate_start(&input)?;
    }
    let _guard = if matches!(action, "start" | "resume") {
        Some(
            relay
                .training_transition
                .try_lock()
                .context("已有训练启动/模式切换正在进行，请等待")?,
        )
    } else {
        None
    };
    request_config(relay, cfg, action, input, None).await
}

async fn request_config(
    relay: &Relay,
    cfg: Config,
    action: &str,
    input: Value,
    operation: Option<&PathBuf>,
) -> anyhow::Result<Value> {
    if input.to_string().len() > 32768 {
        bail!("训练请求过大");
    }
    let client = AutoDl::new(&cfg.autodl)?;
    check_cancelled(operation)?;
    let mut info = client.pick_instance().await?;
    let launching = matches!(action, "start" | "resume");
    if launching
        && needs_gpu_switch(&info)
        && !cfg.autodl.allow_gpu_boot
        && input.get("auto_gpu") != Some(&json!(true))
    {
        bail!(
            "启动需要切换 GPU 计费模式：WebUI 启动按钮会自动切换；CLI 请配置 allow_gpu_boot=true"
        );
    }
    if launching {
        check_cancelled(operation)?;
        progress(
            operation,
            "checking",
            "检查配置、空间和现有训练任务",
            Some(&info.uuid),
        )?;
        if info.is_running() {
            let sftp = connect(relay, &cfg, &info).await?;
            if let Some(id) = input.get("operation_id") {
                let mut lookup = payload(&cfg, &info.uuid, "lookup_operation");
                lookup["operation_id"] = id.clone();
                let found = remote(&sftp, &lookup).await?;
                if found["job"].is_object() { return Ok(found["job"].clone()); }
            }
            let mut check = payload(&cfg, &info.uuid, "preflight");
            if action == "start" {
                check["request"] = input.clone();
                check["gpu_devices"] = input.get("gpu_devices").cloned().unwrap_or(json!(""));
            } else {
                check["id"] = input.get("id").cloned().unwrap_or(Value::Null);
            }
            remote(&sftp, &check).await?;
        }
        info = ensure_gpu(relay, &cfg, &client, info, operation).await?;
    }
    let snapshot = relay
        .data_dir
        .join("training")
        .join(&info.uuid)
        .join("snapshot.json");
    if !info.is_running() {
        if action == "list" || action == "inventory" {
            let mut result = if action == "list" && snapshot.exists() {
                serde_json::from_slice::<Value>(&std::fs::read(&snapshot)?)?
            } else {
                json!({"jobs":[]})
            };
            result["instance"] = info.to_redacted_json();
            result["reachable"] = json!(false);
            result["training"] = serde_json::to_value(&cfg.training)?;
            return Ok(result);
        }
        bail!("实例尚未运行，请先开机");
    }
    let sftp = connect(relay, &cfg, &info).await?;
    let mut body = payload(&cfg, &info.uuid, action);
    if action == "start" {
        body["request"] = input.clone();
        body["gpu_devices"] = input.get("gpu_devices").cloned().unwrap_or(json!(""));
    } else if let Some(object) = input.as_object() {
        for (key, value) in object {
            if ["id", "bytes", "attempt", "operation_id", "shutdown_after_train", "shutdown_delay_seconds"].contains(&key.as_str()) {
                body[key] = value.clone();
            }
        }
    }
    if action == "start" || action == "resume" {
        let mut check = body.clone();
        check["action"] = json!("check_repo");
        let checked = remote(&sftp, &check).await?;
        let root = checked["repo_root"].as_str().context("仓库路径缺失")?;
        sftp.write_file(
            &format!("{root}/.relay_training/manager.py"),
            HELPER.as_bytes(),
        )
        .await?;
    }
    if launching {
        let _control = relay.training_operations_lock.lock().unwrap();
        check_cancelled(operation)?;
        progress(
            operation,
            "starting",
            "GPU 已就绪，正在启动服务器训练",
            None,
        )?;
    }
    let mut result = remote(&sftp, &body).await?;
    result["reachable"] = json!(true);
    result["instance"] = info.to_redacted_json();
    if action == "list" {
        result["observed_at"] = json!(crate::model::now_secs());
        std::fs::create_dir_all(snapshot.parent().unwrap())?;
        let temporary = snapshot.with_extension(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
        std::fs::write(&temporary, serde_json::to_vec_pretty(&result)?)?;
        std::fs::rename(temporary, snapshot)?;
    }
    Ok(result)
}

pub async fn guard_shutdown(
    relay: &Relay,
    instance: &str,
) -> anyhow::Result<tokio::sync::OwnedMutexGuard<()>> {
    let guard = Arc::clone(&relay.training_transition)
        .try_lock_owned()
        .context("训练启动或 GPU 切换中，暂不允许关机")?;
    let mut cfg = relay.config();
    cfg.autodl.instance_uuid = instance.into();
    let info = AutoDl::new(&cfg.autodl)?.pick_instance().await?;
    if !info.is_running() {
        return Ok(guard);
    }
    let sftp = connect(relay, &cfg, &info).await?;
    let mut check=payload(&cfg, instance, "assert_idle");
    // An interrupted code transaction blocks training, but must not force the
    // instance to remain billable once no training/sync process is active.
    check["allow_code_sync"]=json!(true);
    remote(&sftp, &check).await?;
    Ok(guard)
}

pub async fn boot_gpu(relay: &Relay) -> anyhow::Result<Value> {
    let _guard = relay
        .training_transition
        .try_lock()
        .context("已有训练启动/模式切换正在进行")?;
    let cfg = relay.config();
    if !cfg.autodl.allow_gpu_boot {
        bail!("请先启用“允许有卡开机”，有卡模式按 GPU 计费");
    }
    let client = AutoDl::new(&cfg.autodl)?;
    let info = client.pick_instance().await?;
    let ready = ensure_gpu(relay, &cfg, &client, info, None).await?;
    Ok(json!({"ok":true,"message":"有卡实例及 SSH 已就绪","instance_uuid":ready.uuid}))
}

fn needs_gpu_switch(info: &crate::autodl::InstanceInfo) -> bool {
    !info.is_running() || info.start_mode.as_deref() == Some("non_gpu")
}

#[derive(Debug, PartialEq)]
enum GpuStep {
    Shutdown,
    Boot,
    Ready,
    Wait,
}

fn gpu_step(info: &crate::autodl::InstanceInfo, shutdown_sent: bool, boot_sent: bool) -> GpuStep {
    if info.is_running() && info.start_mode.as_deref() == Some("non_gpu") {
        if shutdown_sent || boot_sent {
            GpuStep::Wait
        } else {
            GpuStep::Shutdown
        }
    } else if info.is_off() {
        if boot_sent {
            GpuStep::Wait
        } else {
            GpuStep::Boot
        }
    } else if info.is_running() {
        GpuStep::Ready
    } else {
        GpuStep::Wait
    }
}

async fn ensure_gpu(
    relay: &Relay,
    cfg: &Config,
    client: &AutoDl,
    mut info: crate::autodl::InstanceInfo,
    operation: Option<&PathBuf>,
) -> anyhow::Result<crate::autodl::InstanceInfo> {
    let uuid = info.uuid.clone(); // Pin selection throughout shutdown/boot, even if WebUI changes selection.
    let mut deadline = Instant::now() + Duration::from_secs(cfg.autodl.boot_timeout_secs);
    let (mut shutdown_sent, mut boot_sent) = (false, false);
    // A restart must not send a second power_on while an accepted boot is pending.
    if let Some(path) = operation {
        let record: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if let Some(accepted) = record["boot_accepted_at"].as_u64().filter(|_| !info.is_running()) {
            boot_sent = true;
            let elapsed = (crate::model::now_secs() as u64).saturating_sub(accepted);
            deadline = Instant::now() + Duration::from_secs(cfg.autodl.boot_timeout_secs.saturating_sub(elapsed));
        }
    }
    let mut last_ssh_error = String::new();
    loop {
        check_cancelled(operation)?;
        if let Some(path) = operation {
            let record: Value = serde_json::from_slice(&std::fs::read(path)?)?;
            if let Some(next) = record["next_retry_at"].as_u64() {
                if next > crate::model::now_secs() as u64 {
                    if record["status"] != "waiting_gpu" {
                        progress(operation,"waiting_gpu","恢复 GPU 资源等待队列",Some(&uuid))?;
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    deadline = Instant::now() + Duration::from_secs(cfg.autodl.boot_timeout_secs);
                    continue;
                }
                info = client.instance(&uuid).await?;
                operation_fields(operation,json!({"next_retry_at":null}))?;
                deadline = Instant::now() + Duration::from_secs(cfg.autodl.boot_timeout_secs);
            }
        }
        if Instant::now() >= deadline {
            bail!("等待 GPU/SSH 就绪超时，最后状态 {} {:?}。{} 实例保留当前状态，请检查 AutoDL；可重试启动", info.status, info.start_mode, last_ssh_error);
        }
        match gpu_step(&info, shutdown_sent, boot_sent) {
            GpuStep::Shutdown => {
                let sftp = connect(relay, cfg, &info).await?;
                remote(&sftp, &payload(cfg, &uuid, "assert_idle")).await?;
                // Do not interrupt a relay transfer while switching modes.
                if relay
                    .store
                    .list()
                    .iter()
                    .any(|task| task.status == crate::model::TaskStatus::Uploading)
                {
                    bail!("中继正在上传，请等待交付完成后启动训练");
                }
                progress(
                    operation,
                    "switching_off",
                    "正在关闭无卡实例，随后自动以 GPU 模式开机",
                    Some(&uuid),
                )?;
                client.power_off(&uuid).await?;
                shutdown_sent = true;
            }
            GpuStep::Boot => {
                progress(
                    operation,
                    "booting_gpu",
                    "正在以 GPU 模式开机（按 GPU 计费）",
                    Some(&uuid),
                )?;
                match client.power_on_gpu(&uuid).await {
                    Ok(()) => {
                        boot_sent = true;
                        operation_fields(operation, json!({"boot_accepted_at":crate::model::now_secs(),"next_retry_at":null}))?;
                    }
                    Err(error) if error.downcast_ref::<crate::autodl::GpuUnavailable>().is_some() && operation.is_some() => {
                        record_capacity_wait(operation.unwrap(), cfg.training.gpu_retry_secs, &format!("{error:#}"))?;
                        deadline = Instant::now() + Duration::from_secs(cfg.autodl.boot_timeout_secs);
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            }
            GpuStep::Ready => {
                progress(
                    operation,
                    "waiting_ssh",
                    "有卡实例已运行，等待 SSH 就绪",
                    Some(&uuid),
                )?;
                match connect(relay, cfg, &info).await {
                    Ok(sftp) => {
                        progress(
                            operation,
                            "waiting_cuda",
                            "SSH 已就绪，等待 CUDA 设备可见",
                            Some(&uuid),
                        )?;
                        let probe = remote(&sftp, &payload(cfg, &uuid, "gpu_ready")).await?;
                        if probe["ready"] == json!(true) {
                            return Ok(info);
                        }
                        last_ssh_error = "CUDA 设备尚未可见".into();
                    }
                    Err(error) => last_ssh_error = error.to_string(),
                }
            }
            GpuStep::Wait => {}
        }
        for _ in 0..5 {
            check_cancelled(operation)?;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        info = client.instance(&uuid).await?;
    }
}

fn write_json(path: &PathBuf, value: &Value) -> anyhow::Result<()> {
    std::fs::create_dir_all(path.parent().context("Missing operation directory")?)?;
    let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn operation_fields(path: Option<&PathBuf>, fields: Value) -> anyhow::Result<()> {
    if let Some(path) = path {
        let mut record: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        record.as_object_mut().context("Invalid operation")?.extend(fields.as_object().unwrap().clone());
        record["updated_at"] = json!(crate::model::now_secs());
        write_json(path, &record)?;
    }
    Ok(())
}

fn record_capacity_wait(path: &PathBuf, retry_secs: u64, reason: &str) -> anyhow::Result<()> {
    let record: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    operation_fields(Some(path), json!({"status":"waiting_gpu","message":format!("GPU 暂无空闲资源，已进入训练队列；每 {retry_secs} 秒重试"),
        "next_retry_at":crate::model::now_secs()+retry_secs as i64,"gpu_attempts":record["gpu_attempts"].as_u64().unwrap_or(0)+1,
        "last_gpu_error":reason,"boot_accepted_at":null}))
}

fn progress(
    path: Option<&PathBuf>,
    status: &str,
    message: &str,
    instance: Option<&str>,
) -> anyhow::Result<()> {
    if let Some(path) = path {
        let mut record: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        record["status"] = json!(status);
        record["message"] = json!(message);
        record["updated_at"] = json!(crate::model::now_secs());
        if let Some(instance) = instance {
            record["instance_uuid"] = json!(instance);
        }
        write_json(path, &record)?;
    }
    Ok(())
}

fn operation_records(relay: &Relay) -> anyhow::Result<Vec<Value>> {
    let directory = relay.data_dir.join("training/operations");
    let mut records = Vec::new();
    if directory.exists() {
        for entry in std::fs::read_dir(directory)? {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                records.push(serde_json::from_slice::<Value>(&std::fs::read(path)?)?);
            }
        }
    }
    records.sort_by_key(|record| std::cmp::Reverse(record["created_at"].as_u64().unwrap_or(0)));
    Ok(records)
}

pub fn operations(relay: &Relay) -> anyhow::Result<Value> {
    let mut records=operation_records(relay)?;
    for record in &mut records {
        record.as_object_mut().unwrap().remove("request");
        record.as_object_mut().unwrap().remove("training_snapshot");
    }
    Ok(json!({"operations":records}))
}

fn operation_path(relay: &Relay, id: &str) -> anyhow::Result<PathBuf> {
    if id.len()!=35 || !id.starts_with("op_") || !id[3..].bytes().all(|b|b.is_ascii_hexdigit()) { bail!("排队任务 ID 无效"); }
    Ok(relay.data_dir.join("training/operations").join(format!("{id}.json")))
}

pub fn cancel_operation(relay: &Relay, id: &str) -> anyhow::Result<Value> {
    let _control=relay.training_operations_lock.lock().unwrap();
    let path=operation_path(relay,id)?;
    let record:Value=serde_json::from_slice(&std::fs::read(&path)?)?;
    let status=record["status"].as_str().unwrap_or("");
    if terminal(status) {return Ok(json!({"ok":true,"status":status}));}
    if status=="starting" {bail!("训练命令已下发，请在任务列表暂停或停止训练");}
    std::fs::write(path.with_extension("cancel"), b"cancel")?;
    Ok(json!({"ok":true,"status":"cancelling","message":"已请求取消排队；正在进行的开机请求可能完成，但不会下发训练命令"}))
}

pub fn recover_operations(relay: &Arc<Relay>) -> anyhow::Result<()> {
    for record in operation_records(relay)? {
        if !terminal(record["status"].as_str().unwrap_or("")) {
            let path = relay
                .data_dir
                .join("training/operations")
                .join(format!("{}.json", record["id"].as_str().unwrap()));
            if path.with_extension("cancel").exists() {
                progress(Some(&path),"cancelled","训练排队已取消",None)?;
            } else if record["status"] != "starting" && record["request"].is_object()
                && record["training_snapshot"].is_object() && record["instance_uuid"].as_str().is_some_and(|s|!s.is_empty()) {
                let mut cfg=relay.config();
                cfg.training=serde_json::from_value(record["training_snapshot"].clone())?;
                cfg.autodl.instance_uuid=record["instance_uuid"].as_str().unwrap().into();
                if let Some(mode)=record["api_mode"].as_str(){cfg.autodl.api_mode=mode.into();}
                let action=match record["action"].as_str(){Some("start")=>"start",Some("resume")=>"resume",_=>continue};
                let relay=Arc::clone(relay);
                tokio::spawn(async move {
                    let _guard=Arc::clone(&relay.training_transition).lock_owned().await;
                    run_operation(&relay,cfg,action,record["request"].clone(),path).await;
                });
            } else { progress(
                Some(&path),
                "interrupted",
                "中继重启使启动流程中断；请先刷新任务列表，已有任务保持运行，确认后重试",
                None,
            )?; }
        }
    }
    Ok(())
}

pub async fn submit(
    relay: Arc<Relay>,
    action: &'static str,
    input: Value,
) -> anyhow::Result<Value> {
    if !input.is_object() || input.to_string().len() > 32768 {
        bail!("训练请求必须是 JSON 对象且小于 32 KiB");
    }
    let guard = Arc::clone(&relay.training_transition)
        .try_lock_owned()
        .context("已有训练启动/模式切换正在进行，请等待")?;
    let mut cfg = relay.config();
    if operation_records(&relay)?.iter().any(|record| !terminal(record["status"].as_str().unwrap_or(""))) {
        bail!("已有训练在排队或启动，请等待完成或取消排队");
    }
    let input = if action == "start" {
        crate::training_cache::apply(&cfg, &relay.config_path,
            crate::training_panel::apply(&cfg, &relay.config_path, input)?)?
    } else {
        input
    };
    if action == "start" {
        validate_start(&input)?;
    }
    // Explicit WebUI start is consent for this one GPU transition; it does not change delivery boot defaults.
    if input.get("auto_gpu") == Some(&json!(true)) {
        cfg.autodl.allow_gpu_boot = true;
    }
    let mut input=launch_defaults(input);
    if cfg.autodl.instance_uuid.is_empty() {
        cfg.autodl.instance_uuid=AutoDl::new(&cfg.autodl)?.pick_instance().await?.uuid;
    }
    let id = format!("op_{}", uuid::Uuid::new_v4().simple());
    input["operation_id"]=json!(id);
    let path = relay
        .data_dir
        .join("training/operations")
        .join(format!("{id}.json"));
    let record = json!({"id":id,"action":action,"status":"queued","message":"训练启动已入队，等待 GPU 就绪",
        "instance_uuid":cfg.autodl.instance_uuid,"api_mode":cfg.autodl.api_mode,"created_at":crate::model::now_secs(),
        "request":input,"training_snapshot":cfg.training,"shutdown_after_train":true});
    write_json(&path, &record)?;
    tokio::spawn(async move {
        let _guard = guard;
        run_operation(&relay,cfg,action,input,path).await;
    });
    Ok(json!({"accepted":true,"operation_id":id,"message":"训练启动已入队；GPU 不足时持续等待，成功训练后自动关机"}))
}

async fn run_operation(relay:&Relay,cfg:Config,action:&str,input:Value,path:PathBuf) {
        match request_config(relay, cfg, action, input, Some(&path)).await {
            Ok(job) => {
                let mut record: Value =
                    serde_json::from_slice(&std::fs::read(&path).unwrap_or_default())
                        .unwrap_or(json!({}));
                record["status"] = json!("completed");
                record["message"] = json!("服务器训练已启动");
                record["job_id"] = job["id"].clone();
                record["updated_at"] = json!(crate::model::now_secs());
                if let Err(error) = write_json(&path, &record) {
                    tracing::error!("保存训练启动状态失败：{error}");
                }
            }
            Err(error) => {
                let phase=std::fs::read(&path).ok().and_then(|bytes|serde_json::from_slice::<Value>(&bytes).ok());
                let status=if error.downcast_ref::<QueueCancelled>().is_some(){"cancelled"}
                    else if phase.as_ref().is_some_and(|r|r["status"]=="starting"){"interrupted"}else{"failed"};
                if let Err(save_error) =
                    progress(Some(&path), status, &format!("{error:#}"), None)
                {
                    tracing::error!("保存训练启动失败状态：{save_error}");
                }
            }
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shell_payload_is_literal() {
        assert_eq!(quote("a'b;$(x)"), "'a'\\''b;$(x)'");
    }
    #[test]
    fn relay_launch_always_supplies_shutdown_even_with_stale_unchecked_state() {
        for input in [json!({}),json!({"shutdown_after_train":false,"panel_state":{"shutdown_after_train":false}})] {
            let input=launch_defaults(input);
            assert_eq!(input["shutdown_after_train"],true);
            assert_eq!(input["shutdown_delay_seconds"],60);
        }
    }

    #[tokio::test]
    async fn resource_shortage_waits_past_boot_timeout_then_retries_and_can_cancel() {
        use std::sync::atomic::{AtomicUsize,Ordering};
        let dir=std::env::temp_dir().join(format!("relay-queue-{}",uuid::Uuid::new_v4()));
        let mut cfg=Config::default();cfg.server.data_dir=dir.to_string_lossy().into();
        cfg.autodl.api_mode="pro".into();cfg.autodl.boot_timeout_secs=1;cfg.training.gpu_retry_secs=2;
        let relay=Relay::new(dir.join("config.toml"),cfg.clone()).unwrap();
        let id=format!("op_{}",uuid::Uuid::new_v4().simple());let path=operation_path(&relay,&id).unwrap();
        write_json(&path,&json!({"id":id,"status":"queued","created_at":1})).unwrap();
        let calls=Arc::new(AtomicUsize::new(0));let count=Arc::clone(&calls);
        let app=axum::Router::new().route("/power_on",axum::routing::post(move || {
            let count=Arc::clone(&count); async move {
                let n=count.fetch_add(1,Ordering::SeqCst);
                axum::Json(if n==0 {json!({"code":"InsufficientGPU","msg":"主机空闲 GPU 不足"})}else{json!({"code":"Success","data":null})})
            }
        })).route("/snapshot",axum::routing::get(||async {axum::Json(json!({"code":"Success","data":{"instance_uuid":"fixture","status":"shutdown","start_mode":"non_gpu"}}))}));
        let listener=tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base=format!("http://{}",listener.local_addr().unwrap());
        let server=tokio::spawn(async move{axum::serve(listener,app).await.unwrap()});
        let client=AutoDl::test_client(&cfg.autodl,base);
        let pending_path=path.clone();let pending_relay=Arc::clone(&relay);let pending_cfg=cfg.clone();
        let waiter=tokio::spawn(async move{ensure_gpu(&pending_relay,&pending_cfg,&client,instance("shutdown","non_gpu"),Some(&pending_path)).await});
        tokio::time::timeout(Duration::from_secs(8),async{
            loop{
                let record:Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                if record["boot_accepted_at"].is_number(){break;}
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst),2);
        assert_eq!(operations(&relay).unwrap()["operations"][0]["gpu_attempts"],1);
        cancel_operation(&relay,&id).unwrap();
        let error=waiter.await.unwrap().unwrap_err();assert!(error.downcast_ref::<QueueCancelled>().is_some());
        assert_eq!(calls.load(Ordering::SeqCst),2);
        server.abort();std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn durable_wait_is_restored_cancellable_and_dispatch_is_not_replayed() {
        let dir=std::env::temp_dir().join(format!("relay-recover-{}",uuid::Uuid::new_v4()));
        let mut cfg=Config::default();cfg.server.data_dir=dir.to_string_lossy().into();
        let relay=Relay::new(dir.join("config.toml"),cfg.clone()).unwrap();
        let lock=relay.training_transition.lock().await;
        let id=format!("op_{}",uuid::Uuid::new_v4().simple());let path=operation_path(&relay,&id).unwrap();
        let record=json!({"id":id,"action":"start","status":"waiting_gpu","instance_uuid":"pinned-instance","created_at":1,
            "request":{"auto_gpu":true,"shutdown_after_train":true},"training_snapshot":cfg.training});
        write_json(&path,&record).unwrap();
        recover_operations(&relay).unwrap();
        let public=operations(&relay).unwrap();assert_eq!(public["operations"][0]["status"],"waiting_gpu");
        assert!(public["operations"][0].get("request").is_none());
        cancel_operation(&relay,&id).unwrap();drop(lock);
        tokio::time::timeout(Duration::from_secs(2),async{
            while operations(&relay).unwrap()["operations"][0]["status"]!="cancelled" {tokio::time::sleep(Duration::from_millis(10)).await;}
        }).await.unwrap();
        let mut dispatched=record;dispatched["status"]=json!("starting");write_json(&path,&dispatched).unwrap();
        std::fs::remove_file(path.with_extension("cancel")).unwrap();
        assert!(cancel_operation(&relay,&id).is_err());
        recover_operations(&relay).unwrap();
        assert_eq!(operations(&relay).unwrap()["operations"][0]["status"],"interrupted");
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn invalid_dataset_is_rejected_and_no_identity_review_is_required() {
        assert!(validate_start(&json!({"identity_reviewed":false})).is_err());
        assert!(validate_start(&json!({"args":{"data_path":"dataset/example.parquet"}})).is_ok());
        assert!(validate_start(&json!({"identity_reviewed":true,"audit_path":"report.json","args":{"data_path":"dataset/pretrain_coder_12b.parquet"}})).is_ok());
        assert!(validate_start(&json!({"identity_reviewed":true,"audit_path":"report.json","args":{"data_path":"dataset/pretrain_identity_clean.parquet"}})).is_ok());
        assert!(validate_start(&json!({"identity_reviewed":true,"audit_path":"report.json","args":{"data_path":"dataset/pretrain_coder_12b.txt"}})).is_err());
    }
    fn instance(status: &str, mode: &str) -> crate::autodl::InstanceInfo {
        crate::autodl::InstanceInfo {
            uuid: "fixture".into(),
            name: None,
            status: status.into(),
            start_mode: Some(mode.into()),
            charge_type: None,
            ssh: None,
            jupyter: None,
            data_disk_total: None,
            data_disk_used: None,
            raw: json!({}),
        }
    }
    #[test]
    fn no_gpu_transition_waits_for_shutdown_and_never_repeats_requests() {
        let running = instance("running", "non_gpu");
        assert_eq!(gpu_step(&running, false, false), GpuStep::Shutdown);
        assert_eq!(gpu_step(&running, true, false), GpuStep::Wait);
        assert_eq!(
            gpu_step(&instance("stopping", "non_gpu"), true, false),
            GpuStep::Wait
        );
        assert_eq!(
            gpu_step(&instance("shutdown", "non_gpu"), true, false),
            GpuStep::Boot
        );
        assert_eq!(
            gpu_step(&instance("shutdown", "non_gpu"), true, true),
            GpuStep::Wait
        );
        assert_eq!(gpu_step(&running, true, true), GpuStep::Wait);
        assert_eq!(
            gpu_step(&instance("running", "gpu"), true, true),
            GpuStep::Ready
        );
    }
    #[test]
    fn already_gpu_instance_is_never_rebooted() {
        let info = instance("running", "gpu");
        assert!(!needs_gpu_switch(&info));
        assert_eq!(gpu_step(&info, false, false), GpuStep::Ready);
        assert_eq!(
            gpu_step(&instance("stopped", "gpu"), false, false),
            GpuStep::Boot
        );
    }
    #[tokio::test]
    async fn duplicate_start_is_blocked_and_pending_operation_survives_as_interrupted() {
        let directory =
            std::env::temp_dir().join(format!("relay-training-{}", uuid::Uuid::new_v4()));
        let mut cfg = Config::default();
        cfg.server.data_dir = directory.to_string_lossy().into();
        let relay = Relay::new(directory.join("config.toml"), cfg).unwrap();
        let lock = relay.training_transition.lock().await;
        assert!(
            submit(Arc::clone(&relay), "start", json!({"auto_gpu":true}))
                .await
                .is_err()
        );
        assert!(guard_shutdown(&relay, "fixture").await.is_err());
        drop(lock);
        let path = directory.join("training/operations/op_fixture.json");
        write_json(
            &path,
            &json!({"id":"op_fixture","created_at":1,"status":"waiting_cuda"}),
        )
        .unwrap();
        recover_operations(&relay).unwrap();
        assert_eq!(
            operations(&relay).unwrap()["operations"][0]["status"],
            "interrupted"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
