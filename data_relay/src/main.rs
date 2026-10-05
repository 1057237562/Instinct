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
mod disk;
mod jupyter;
mod model;
mod pull;
mod runner;
mod sftp;
mod store;
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
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" => {
                config_path = PathBuf::from(args.next().unwrap_or_else(|| {
                    eprintln!("--config 需要路径参数");
                    std::process::exit(2);
                }));
            }
            "serve" | "check" => cmd = arg,
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
    let cfg = Config::load(&config_path)?;
    let relay = Relay::new(config_path, cfg)?;

    match cmd.as_str() {
        "check" => check(&relay).await,
        _ => serve(relay).await,
    }
}

async fn serve(relay: Arc<Relay>) -> anyhow::Result<()> {
    let listen = relay.config().server.listen.clone();
    let app = web::router(Arc::clone(&relay));
    let listener = tokio::net::TcpListener::bind(&listen).await?;
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
