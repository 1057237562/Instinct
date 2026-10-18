//! Snapshot the local working copy, compare against a pinned instance, then sync reviewed bytes.
use crate::{
    autodl::AutoDl,
    config::Config,
    model,
    runner::Relay,
    sftp::{self, SftpUploader},
};
use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

const HELPER: &str = include_str!("code_sync_remote.py");
const GUARD: &str = include_str!("training_remote.py");

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileRecord {
    path: String,
    size: u64,
    sha256: String,
    #[serde(default)]
    remote_sha256: Option<String>,
    #[serde(default)]
    status: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Plan {
    id: String,
    instance_uuid: String,
    local_root: String,
    remote_root: String,
    include_assets: bool,
    created_at: i64,
    updated_at: i64,
    status: String,
    message: String,
    files: Vec<FileRecord>,
    total_bytes: u64,
    changed_bytes: u64,
    copied_bytes: u64,
    guard_sha256: String,
    backup_path: String,
}
#[derive(Deserialize)]
pub struct CompareRequest {
    instance_uuid: String,
    #[serde(default = "default_assets")]
    include_assets: bool,
}
fn default_assets() -> bool {
    true
}

pub fn allowed(relative: &str, assets: bool) -> bool {
    let parts: Vec<_> = relative.split('/').collect();
    if relative.is_empty()
        || relative.contains('\\')
        || relative.chars().any(char::is_control)
        || parts
            .iter()
            .any(|p| p.is_empty() || p.starts_with('.') || *p == "__pycache__")
    {
        return false;
    }
    let name = parts[parts.len() - 1];
    if name.ends_with(".py") {
        return (parts.len() > 1 && matches!(parts[0], "trainer" | "model"))
            || (parts.len() > 1
                && parts[0] == "scripts"
                && (parts.len() == 2 || parts[1] == "data_loader"))
            || matches!(relative, "run_loop.py" | "run_linear.py" | "eval_llm.py");
    }
    if matches!(
        relative,
        "start_config_webui.bat" | "scripts/monitor_training_shutdown.ps1"
    ) {
        return true;
    }
    assets
        && ((parts.len() == 1 && name.starts_with("requirements") && name.ends_with(".txt"))
            || (parts.len() == 2
                && parts[0] == "trainer"
                && ((name.starts_with("config_") && name.ends_with(".json"))
                    || name == "autotune_pretrain.json"))
            || matches!(
                relative,
                "model/tokenizer.json"
                    | "model/tokenizer_config.json"
                    | "model/special_tokens_map.json"
                    | "model/vocab.json"
                    | "model/merges.txt"
            ))
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn valid_id(id: &str) -> bool {
    id.starts_with("cs_") && id.len() == 35 && id[3..].bytes().all(|c| c.is_ascii_hexdigit())
}
fn directory(relay: &Relay, id: &str) -> anyhow::Result<PathBuf> {
    if !valid_id(id) {
        bail!("同步计划 ID 无效");
    }
    Ok(relay.data_dir.join("code_sync").join(id))
}
fn save(relay: &Relay, plan: &Plan) -> anyhow::Result<()> {
    let dir = directory(relay, &plan.id)?;
    std::fs::create_dir_all(&dir)?;
    let temporary = dir.join(format!("plan-{}.tmp", uuid::Uuid::new_v4()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(plan)?)?;
    std::fs::rename(temporary, dir.join("plan.json"))?;
    Ok(())
}
pub fn get(relay: &Relay, id: &str) -> anyhow::Result<Plan> {
    Ok(serde_json::from_slice(&std::fs::read(
        directory(relay, id)?.join("plan.json"),
    )?)?)
}
pub fn list(relay: &Relay) -> anyhow::Result<Value> {
    let root = relay.data_dir.join("code_sync");
    let mut plans = Vec::new();
    if root.exists() {
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            let id = entry.file_name().to_string_lossy().to_string();
            if valid_id(&id) {
                if let Ok(plan) = get(relay, &id) {
                    plans.push(plan);
                }
            }
        }
    }
    plans.sort_by_key(|p| std::cmp::Reverse(p.created_at));
    plans.truncate(20);
    let rows:Vec<_>=plans.into_iter().map(|p|json!({"id":p.id,"instance_uuid":p.instance_uuid,"status":p.status,
        "message":p.message,"created_at":p.created_at,"remote_root":p.remote_root,"backup_path":p.backup_path})).collect();
    Ok(json!({"plans":rows}))
}
pub fn recover(relay: &Relay) -> anyhow::Result<()> {
    let root = relay.data_dir.join("code_sync");
    if !root.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(root)? {
        let id = entry?.file_name().to_string_lossy().to_string();
        if !valid_id(&id) {
            continue;
        }
        let Ok(mut p) = get(relay, &id) else {
            continue;
        };
        if matches!(p.status.as_str(), "comparing" | "syncing" | "rolling_back") {
            p.status = "interrupted".into();
            p.message = "中继重启使操作中断；请检查远端或回滚后重新比较".into();
            save(relay, &p)?;
        }
    }
    Ok(())
}
fn source_root(relay: &Relay, cfg: &Config) -> anyhow::Result<PathBuf> {
    let configured = Path::new(&cfg.training.local_repo_root);
    let root = if configured.is_absolute() {
        configured.to_path_buf()
    } else {
        relay
            .config_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(configured)
    }
    .canonicalize()?;
    for marker in [
        "scripts/config_webui.py",
        "model/model_instinct.py",
        "trainer/train_pretrain.py",
    ] {
        if !root.join(marker).is_file() {
            bail!("本地训练仓库缺少 {marker}：{}", root.display());
        }
    }
    Ok(root)
}
fn snapshot(root: &Path, destination: &Path, assets: bool) -> anyhow::Result<Vec<FileRecord>> {
    let mut candidates = Vec::new();
    for item in std::fs::read_dir(root)? {
        let item = item?;
        if item.file_type()?.is_file() {
            candidates.push(item.path());
        }
    }
    for folder in ["trainer", "model", "scripts"] {
        if root
            .join(folder)
            .symlink_metadata()?
            .file_type()
            .is_symlink()
        {
            bail!("同步目录不能是符号链接：{folder}");
        }
        for entry in walkdir::WalkDir::new(root.join(folder))
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                if e.depth() == 0 {
                    return true;
                }
                let rel = e
                    .path()
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                if e.file_type().is_dir() {
                    !e.file_name().to_string_lossy().starts_with('.')
                        && e.file_name() != "__pycache__"
                        && (!rel.starts_with("scripts/")
                            || rel == "scripts/data_loader"
                            || rel.starts_with("scripts/data_loader/"))
                } else {
                    true
                }
            })
        {
            let entry = entry?;
            if entry.file_type().is_file() || entry.file_type().is_symlink() {
                candidates.push(entry.path().to_path_buf());
            }
        }
    }
    let mut records = Vec::new();
    let mut total = 0;
    let mut names = std::collections::HashSet::new();
    for path in candidates {
        let rel = path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        if !allowed(&rel, assets) {
            continue;
        }
        if path.symlink_metadata()?.file_type().is_symlink()
            || !path.canonicalize()?.starts_with(root)
        {
            bail!("同步路径不能是符号链接：{rel}");
        }
        if !names.insert(rel.to_lowercase()) {
            bail!("重复的同步路径：{rel}");
        }
        if path.metadata()?.len() > 32 * 1024 * 1024 {
            bail!("代码文件超过 32 MiB：{rel}");
        }
        let bytes = std::fs::read(&path)?;
        total += bytes.len();
        if total > 128 * 1024 * 1024 || records.len() >= 4096 {
            bail!("训练代码快照超过大小或文件数限制");
        }
        let out = destination.join(&rel);
        std::fs::create_dir_all(out.parent().unwrap())?;
        std::fs::write(out, &bytes)?;
        records.push(FileRecord {
            path: rel,
            size: bytes.len() as u64,
            sha256: hash(&bytes),
            remote_sha256: None,
            status: String::new(),
        });
    }
    records.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(records)
}
async fn remote(connection: &SftpUploader, plan: &Plan, action: &str) -> anyhow::Result<Value> {
    let args = json!({"id":plan.id,"repo_root":plan.remote_root,"action":action,"guard_sha256":plan.guard_sha256});
    let (code, output) = connection
        .exec(&sftp::python_command(HELPER, &args.to_string()))
        .await?;
    if code != 0 {
        bail!("代码同步失败：{}", output.trim());
    }
    serde_json::from_str(output.trim()).context("远端代码同步响应不是 JSON")
}
async fn connect(relay: &Relay, plan: &Plan, cfg: &Config) -> anyhow::Result<SftpUploader> {
    let mut cfg = cfg.clone();
    cfg.autodl.instance_uuid = plan.instance_uuid.clone();
    let info = AutoDl::new(&cfg.autodl)?.pick_instance().await?;
    if !info.is_running() {
        bail!("目标实例未运行，请先无卡开机；比较和同步不会自动开机");
    }
    crate::training::connect(relay, &cfg, &info).await
}
pub async fn preview(relay: &Relay, input: CompareRequest) -> anyhow::Result<Plan> {
    if input.instance_uuid.trim().is_empty() || input.instance_uuid.len() > 200 {
        bail!("请选择目标实例");
    }
    let cfg = relay.config();
    let root = source_root(relay, &cfg)?;
    let id = format!("cs_{}", uuid::Uuid::new_v4().simple());
    let files = snapshot(
        &root,
        &directory(relay, &id)?.join("files"),
        input.include_assets,
    )?;
    let mut plan = Plan {
        id,
        instance_uuid: input.instance_uuid,
        local_root: root.to_string_lossy().into(),
        remote_root: cfg.training.repo_root.clone(),
        include_assets: input.include_assets,
        created_at: model::now_secs(),
        updated_at: model::now_secs(),
        status: "comparing".into(),
        message: "正在比较远端文件".into(),
        total_bytes: files.iter().map(|f| f.size).sum(),
        changed_bytes: 0,
        copied_bytes: 0,
        files,
        guard_sha256: hash(GUARD.as_bytes()),
        backup_path: String::new(),
    };
    save(relay, &plan)?;
    let result = async {
        let connection = connect(relay, &plan, &cfg).await?;
        remote(&connection, &plan, "prepare").await?;
        let base = format!("{}/.relay_code_sync/{}", plan.remote_root, plan.id);
        connection
            .write_file(
                &format!("{base}/manifest.json"),
                json!({"files":plan.files}).to_string().as_bytes(),
            )
            .await?;
        connection
            .write_file(&format!("{base}/training_guard.py"), GUARD.as_bytes())
            .await?;
        remote(&connection, &plan, "preview").await
    }
    .await;
    match result {
        Ok(value) => {
            let compared: Vec<FileRecord> = serde_json::from_value(value["files"].clone())?;
            if compared.len() != plan.files.len()
                || compared
                    .iter()
                    .zip(&plan.files)
                    .any(|(a, b)| a.path != b.path || a.sha256 != b.sha256 || a.size != b.size)
            {
                bail!("远端返回的清单与本地快照不一致");
            }
            plan.files = compared;
            plan.changed_bytes = plan
                .files
                .iter()
                .filter(|f| f.status != "same")
                .map(|f| f.size)
                .sum();
            plan.backup_path = value["backup_path"].as_str().unwrap_or("").into();
            plan.status = "ready".into();
            plan.message = "比较完成，请核对变更后同步".into();
            save(relay, &plan)?;
            Ok(plan)
        }
        Err(error) => {
            plan.status = "failed".into();
            plan.message = format!("{error:#}");
            save(relay, &plan)?;
            Err(error)
        }
    }
}
pub async fn submit(relay: Arc<Relay>, id: String, rollback: bool) -> anyhow::Result<Plan> {
    let guard = relay
        .training_transition
        .clone()
        .try_lock_owned()
        .context("正在训练切换、交付或同步，请稍后重试")?;
    let mut plan = get(&relay, &id)?;
    if rollback {
        if !matches!(plan.status.as_str(), "completed" | "failed" | "interrupted") {
            bail!("此计划当前不可回滚");
        }
    } else if plan.status != "ready" {
        bail!("请先重新比较代码差异");
    }
    let cfg = relay.config();
    if !rollback && plan.remote_root != cfg.training.repo_root {
        bail!("远端仓库设置已变化，请重新比较");
    }
    plan.status = if rollback { "rolling_back" } else { "syncing" }.into();
    plan.message = if rollback {
        "正在恢复远端备份"
    } else {
        "正在上传变更文件并验证"
    }
    .into();
    save(&relay, &plan)?;
    let response = plan.clone();
    tokio::spawn(async move {
        let _guard = guard;
        let result = async {
            let connection = connect(&relay, &plan, &cfg).await?;
            if !rollback {
                crate::training::assert_idle(&connection, &cfg, &plan.instance_uuid).await?;
            }
            if !rollback {
                for file in plan.files.clone().iter().filter(|f| f.status != "same") {
                    if !allowed(&file.path, plan.include_assets) {
                        bail!("计划包含禁止同步的路径");
                    }
                    let bytes =
                        std::fs::read(directory(&relay, &id)?.join("files").join(&file.path))?;
                    if bytes.len() as u64 != file.size || hash(&bytes) != file.sha256 {
                        bail!("本地快照已变化，请重新比较：{}", file.path);
                    }
                    let remote_path = format!(
                        "{}/.relay_code_sync/{}/files/{}",
                        plan.remote_root, id, file.path
                    );
                    connection
                        .mkdir_p(remote_path.rsplit_once('/').unwrap().0)
                        .await?;
                    connection.write_file(&remote_path, &bytes).await?;
                    plan.copied_bytes += file.size;
                    plan.updated_at = model::now_secs();
                    save(&relay, &plan)?;
                }
            }
            remote(
                &connection,
                &plan,
                if rollback { "rollback" } else { "apply" },
            )
            .await
        }
        .await;
        match result {
            Ok(value) => {
                plan.status = value["status"].as_str().unwrap_or("completed").into();
                plan.message = if rollback {
                    "远端备份已恢复"
                } else {
                    "同步完成，全部文件 SHA-256 校验通过"
                }
                .into();
            }
            Err(error) => {
                plan.status = "failed".into();
                plan.message = format!("{error:#}");
            }
        }
        plan.updated_at = model::now_secs();
        save(&relay, &plan).ok();
    });
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scope_excludes_runtime_data_and_keeps_training_dependencies() {
        for path in [
            "trainer/train_pretrain.py",
            "model/moe_dispatch.py",
            "scripts/config_webui.py",
            "scripts/data_loader/lm_dataset.py",
            "trainer/config_pretrain.json",
            "model/tokenizer.json",
            "requirements.txt",
        ] {
            assert!(allowed(path, true), "{path}");
        }
        for path in [
            "trainer/webui_state.json",
            "dataset/data.parquet",
            "out/model.pth",
            "checkpoints/x.pth",
            ".env",
            "data_relay/config.toml",
            "scripts/data_builder/build.py",
            "trainer/../out/x.py",
            "scripts/__pycache__/x.pyc",
        ] {
            assert!(!allowed(path, true), "{path}");
        }
        assert!(!allowed("trainer/config_pretrain.json", false));
        assert!(allowed("trainer/train_pretrain.py", false));
    }
    #[test]
    fn real_checkout_scope_covers_config_webui_and_loader() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .canonicalize()
            .unwrap();
        let dir =
            std::env::temp_dir().join(format!("relay_code_snapshot_{}", uuid::Uuid::new_v4()));
        let records = snapshot(&root, &dir, true).unwrap();
        for path in [
            "scripts/config_webui.py",
            "scripts/config_layout.py",
            "scripts/autotune_panel.py",
            "scripts/pipeline_panel.py",
            "scripts/sft_profiles.py",
            "trainer/trainer_utils.py",
            "trainer/compile_cache.py",
            "scripts/data_loader/source_format.py",
            "run_loop.py",
        ] {
            assert!(records.iter().any(|f| f.path == path), "{path}");
        }
        assert!(records.iter().all(|f| allowed(&f.path, true)));
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[tokio::test]
    async fn api_auth_and_transition_guard_precede_any_remote_mutation() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;
        let dir = std::env::temp_dir().join(format!("relay_code_api_{}", uuid::Uuid::new_v4()));
        let mut cfg = Config::default();
        cfg.server.data_dir = dir.to_string_lossy().into();
        cfg.server.api_token = "test".into();
        let relay = Relay::new(dir.join("config.toml"), cfg).unwrap();
        let app = crate::web::router(relay.clone());
        for path in [
            "/api/code-sync",
            "/api/code-sync/cs_bad/apply",
            "/api/code-sync/cs_bad/rollback",
        ] {
            let result = app
                .clone()
                .oneshot(Request::post(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(result.status(), 401);
        }
        let _guard = relay.training_transition.lock().await;
        let result = app
            .oneshot(
                Request::post("/api/code-sync/cs_bad/apply")
                    .header("authorization", "test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.status(), 400);
        let body = axum::body::to_bytes(result.into_body(), 4096)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("请稍后重试"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
