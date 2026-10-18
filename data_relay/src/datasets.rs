//! Fixed-root remote dataset browser. Listing never boots an instance.
use crate::{autodl::AutoDl, model::TaskStatus, runner::Relay, sftp};
use anyhow::{bail, Context};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Default, Deserialize)]
pub struct Request {
    #[serde(default)]
    pub instance_uuid: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub version: String,
}

fn validate(input: &Request, action: &str) -> anyhow::Result<()> {
    if input.instance_uuid.is_empty() || input.instance_uuid.len() > 200 {
        bail!("请先选择目标实例");
    }
    if input.path.len() > 4096 || input.path.contains('\\') || input.path.chars().any(char::is_control)
        || (!input.path.is_empty() && input.path.split('/').any(|p| p.is_empty() || p.starts_with('.')))
    {
        bail!("路径必须位于 dataset 内，不能包含隐藏目录或路径跳转");
    }
    if action != "list" && input.path.is_empty() { bail!("请选择具体文件"); }
    if action == "delete" && (input.version.len() != 64 || !input.version.bytes().all(|c| c.is_ascii_hexdigit())) {
        bail!("文件版本缺失，请刷新后重试");
    }
    Ok(())
}

pub async fn request(relay: &Relay, action: &str, input: Request) -> anyhow::Result<Value> {
    validate(&input, action)?;
    let deleting = action == "delete";
    let _guard = if deleting {
        Some(relay.training_transition.try_lock().context("数据正在交付或训练正在切换，暂不能删除")?)
    } else { None };
    if deleting && relay.store.list().iter().any(|t| matches!(t.status, TaskStatus::Uploading | TaskStatus::Verifying)) {
        bail!("有数据正在校验或交付，完成后再删除");
    }
    let mut cfg = relay.config();
    cfg.autodl.instance_uuid = input.instance_uuid.clone();
    let info = AutoDl::new(&cfg.autodl)?.pick_instance().await?;
    if !info.is_running() {
        if action == "list" {
            return Ok(json!({"instance_uuid":info.uuid,"reachable":false,"status":info.status,
                "root":crate::data_upload::DATASET_ROOT,"path":input.path,"entries":[]}));
        }
        bail!("实例未运行，请先开机");
    }
    let connection = crate::training::connect(relay, &cfg, &info).await?;
    if deleting { crate::training::assert_idle(&connection, &cfg, &info.uuid).await?; }
    let payload = json!({"action":action,"path":input.path,"version":input.version});
    let command = sftp::python_command(include_str!("datasets_remote.py"), &payload.to_string());
    let (code, output) = connection.exec(&command).await?;
    if code != 0 { bail!("dataset 操作失败：{}", output.trim()); }
    let mut value: Value = serde_json::from_str(output.trim()).context("dataset 响应无效")?;
    value["instance_uuid"] = json!(info.uuid);
    value["reachable"] = json!(true);
    value["status"] = json!(info.status);
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paths_and_delete_versions_are_checked_before_network() {
        for path in ["/etc/passwd", "../a", "a/../b", "a//b", "a\\b", ".relay_incoming/x", "a/.secret", "a\nb"] {
            assert!(validate(&Request {instance_uuid:"test".into(),path:path.into(),..Default::default()}, "list").is_err());
        }
        let mut input = Request {instance_uuid:"test".into(),path:"shards/训练.parquet".into(),..Default::default()};
        assert!(validate(&input,"list").is_ok());
        assert!(validate(&input,"delete").is_err());
        input.version="a".repeat(64);
        assert!(validate(&input,"delete").is_ok());
        input.path.clear();
        assert!(validate(&input,"list").is_ok());
        assert!(validate(&input,"delete").is_err());
    }

    #[tokio::test]
    async fn api_requires_auth_and_blocks_delete_during_delivery() {
        use axum::{body::Body, http::Request as HttpRequest};
        use tower::ServiceExt;
        let dir = std::env::temp_dir().join(format!("relay_datasets_{}",uuid::Uuid::new_v4()));
        let mut cfg = crate::config::Config::default();
        cfg.server.data_dir=dir.to_string_lossy().into();cfg.server.api_token="test-token".into();
        let relay=Relay::new(dir.join("config.toml"),cfg).unwrap();
        let app=crate::web::router(relay.clone());
        for (method,path) in [("GET","/api/datasets"),("GET","/api/datasets/preview"),("DELETE","/api/datasets")] {
            let result=app.clone().oneshot(HttpRequest::builder().method(method).uri(path).body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(result.status(),401);
        }
        let _guard=relay.training_transition.lock().await;
        let body=json!({"instance_uuid":"test","path":"corpus.parquet","version":"a".repeat(64)});
        let result=app.oneshot(HttpRequest::delete("/api/datasets").header("authorization","test-token")
            .header("content-type","application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
        assert_eq!(result.status(),400);
        let bytes=axum::body::to_bytes(result.into_body(),4096).await.unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("暂不能删除"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
