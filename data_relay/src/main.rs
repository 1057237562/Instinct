//! data_relay — 数据下载中继服务器。
//!
//! 流程：上传下载脚本 → 自动执行 → 完成后 AutoDL 无卡模式开机 →
//! 数据上传（SFTP 主通道 / Jupyter 备用通道）→ 数据盘配额管理 → 自动关机。
//!
//! 子命令：
//!   serve   启动 WebUI/API（默认）
//!   check   检查 AutoDL 连通性与数据盘用量后退出

mod autodl;
mod config;
mod code_sync;
mod data_upload;
mod datasets;
mod disk;
mod deliver;
mod jupyter;
mod model;
mod pull;
mod runner;
mod sftp;
mod store;
mod training;
mod training_cache;
mod training_panel;
mod web;

use config::Config;
use runner::Relay;
use std::path::PathBuf;
use std::sync::Arc;

const CONFIG_EXAMPLE: &str = include_str!("../config.example.toml");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,data_relay=debug".into()),
        )
        .with_target(false)
        .init();

    let mut args = std::env::args().skip(1);
    let mut cmd = "serve".to_string();
    let mut config_path = PathBuf::from("config.toml");
    let mut delivery = deliver::Options::default();
    let mut instance_override = None;
    let mut remote_script = None;
    let mut training_action = "list".to_string();
    let mut training_job = None;
    let mut training_request = None;
    let mut auto_gpu = false;
    let mut from_panel_cache = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                config_path = PathBuf::from(args.next().unwrap_or_else(|| {
                    eprintln!("--config 需要路径参数");
                    std::process::exit(2);
                }));
            }
            "serve" | "check" | "deliver" | "inspect-server" | "remote-script" | "training" => cmd = arg,
            "--training-action" => training_action = args.next().ok_or_else(|| anyhow::anyhow!("--training-action requires an action"))?,
            "--job-id" => training_job = Some(args.next().ok_or_else(|| anyhow::anyhow!("--job-id requires an ID"))?),
            "--request-json" => training_request = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("--request-json requires a path"))?)),
            "--auto-gpu" => auto_gpu = true,
            "--from-panel-cache" => from_panel_cache = true,
            "--remote-script" => remote_script = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("--remote-script requires a local Python file"))?)),
            "--source" => delivery.source = PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("--source requires a path"))?),
            "--repo" => delivery.repo = args.next().ok_or_else(|| anyhow::anyhow!("--repo requires an absolute remote path"))?,
            "--remove-old" => delivery.remove_old.push(args.next().ok_or_else(|| anyhow::anyhow!("--remove-old requires a dataset-relative path"))?),
            "--execute" => delivery.execute = true,
            "--skip-git-update" => delivery.skip_git_update = true,
            "--init-repo" => delivery.init_repo = true,
            "--clean-cache" => delivery.clean_cache = true,
            "--git-bundle" => delivery.git_bundle = Some(PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("--git-bundle requires a path"))?)),
            "--branch" => delivery.branch = args.next().ok_or_else(|| anyhow::anyhow!("--branch requires a branch name"))?,
            "--instance" => instance_override = Some(args.next().ok_or_else(|| anyhow::anyhow!("--instance requires a UUID"))?),
            "--help" | "-h" => {
                println!("data_relay [--config config.toml] [--instance UUID] [serve|check|inspect-server|deliver|remote-script|training]\ntraining --training-action panel_cache|list|inventory|start|get|log|pause|resume|stop|boot_gpu [--job-id ID] [--request-json FILE] [--from-panel-cache] [--auto-gpu]\n--auto-gpu: consent to shutdown no-GPU mode and boot GPU for this training request (GPU billing).\ninspect-server: inspect remote code/data; may boot in no-GPU mode.\nremote-script --remote-script LOCAL_PYTHON_FILE: run an explicit maintenance script on the instance.\ndeliver --source FILE --repo /absolute/Instinct [--branch BRANCH] [--init-repo]\n        [--remove-old FILE_OR_DIR]... [--clean-cache] [--git-bundle FILE] [--skip-git-update] [--execute]\nWithout --execute, deliver only inspects a running instance; no boot, deletion, git update or upload.");
                return Ok(());
            }
            other => {
                eprintln!("未知参数: {other}（用法: data_relay [--config config.toml] [serve|check]）");
                std::process::exit(2);
            }
        }
    }

    if !config_path.exists() {
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&config_path, CONFIG_EXAMPLE)?;
        println!(
            "已生成默认配置 {}：请填写 autodl.console_token（浏览器控制台 LocalStorage 的 token）后重新启动。",
            config_path.display()
        );
        return Ok(());
    }
    let mut cfg = Config::load(&config_path)?;
    if let Some(uuid) = instance_override { cfg.autodl.instance_uuid = uuid; }
    if (auto_gpu || from_panel_cache) && cmd != "training" { anyhow::bail!("--auto-gpu/--from-panel-cache are only valid for training"); }
    if auto_gpu { cfg.autodl.allow_gpu_boot = true; }
    if cmd == "remote-script" {
        return deliver::run_remote_script(&cfg, remote_script.as_deref().ok_or_else(|| anyhow::anyhow!("remote-script requires --remote-script FILE"))?).await;
    }
    if cmd == "inspect-server" {
        return deliver::inspect_server(&cfg).await;
    }
    if cmd == "deliver" {
        return deliver::run(&cfg, delivery).await;
    }
    let relay = Relay::new(config_path, cfg)?;
    if cmd == "training" {
        let mut input = if let Some(path) = training_request { serde_json::from_slice(&std::fs::read(path)?)? } else { serde_json::json!({}) };
        if !input.is_object() { anyhow::bail!("Training request JSON must be an object"); }
        if let Some(id) = training_job { input["id"] = serde_json::json!(id); }
        if from_panel_cache { input["use_panel_cache"] = serde_json::json!(true); }
        if !["panel_cache", "list", "inventory", "start", "get", "log", "pause", "resume", "stop", "boot_gpu"].contains(&training_action.as_str()) {
            anyhow::bail!("Unknown training action: {training_action}");
        }
        let result = if training_action == "panel_cache" { training_cache::load(&relay.config(), &relay.config_path, input.get("trainer").and_then(|value| value.as_str()))? }
            else if training_action == "boot_gpu" { training::boot_gpu(&relay).await? }
            else { training::request(&relay, &training_action, input).await? };
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    match cmd.as_str() {
        "check" => check(&relay).await,
        _ => serve(relay).await,
    }
}

async fn serve(relay: Arc<Relay>) -> anyhow::Result<()> {
    let listen = relay.config().server.listen.clone();
    let app = web::router(Arc::clone(&relay));
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    training::recover_operations(&relay)?;
    code_sync::recover(&relay)?;
    tracing::info!("data_relay 监听 http://{listen}（WebUI: http://{listen}/）");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("收到退出信号，正在关闭…");
        })
        .await?;
    Ok(())
}

async fn check(relay: &Arc<Relay>) -> anyhow::Result<()> {
    let cfg = relay.config();
    println!(
        "api_mode={} instance={:?}",
        cfg.autodl.api_mode, cfg.autodl.instance_uuid
    );
    let client = autodl::AutoDl::new(&cfg.autodl)?;
    let info = client.pick_instance().await?;
    println!(
        "实例 {} 状态={} start_mode={:?} name={:?}",
        info.uuid, info.status, info.start_mode, info.name
    );
    if let (Some(total), Some(used)) = (info.data_disk_total, info.data_disk_used) {
        println!(
            "数据盘(API): 已用 {} / 总量 {}（剩 {}）",
            model::human_bytes(used),
            model::human_bytes(total),
            model::human_bytes(total.saturating_sub(used))
        );
    }
    let report = relay.autodl_disk().await?;
    println!("磁盘报告: {}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
