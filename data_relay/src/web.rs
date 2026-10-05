//! Axum HTTP layer: WebUI + JSON API. Every /api route requires the relay
//! token (`Authorization: <token>` or `Bearer <token>`).

use crate::model::UploadMode;
use crate::runner::Relay;
use axum::extract::{Multipart, Path as AxumPath, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{Value, json};
use std::sync::Arc;

const INDEX_HTML: &str = include_str!("index.html");

pub fn router(state: Arc<Relay>) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/tasks", get(list_tasks).post(create_task))
        .route("/api/tasks/{id}", get(get_task).delete(delete_task))
        .route("/api/tasks/{id}/log", get(task_log))
        .route("/api/tasks/{id}/start", post(start_task))
        .route("/api/tasks/{id}/upload", post(upload_task))
        .route("/api/tasks/{id}/cancel", post(cancel_task))
        .route("/api/autodl/status", get(autodl_status))
        .route("/api/autodl/instances", get(list_instances))
        .route("/api/autodl/boot_no_gpu", post(boot_no_gpu))
        .route("/api/autodl/power_off", post(power_off))
        .route("/api/autodl/wx_start", post(wx_login_start))
        .route("/api/autodl/wx_poll", post(wx_login_poll))
        .route("/api/autodl/disk", get(autodl_disk))
        .route("/api/autodl/remote", get(list_remote).delete(delete_remote))
        .route("/api/config", get(get_config).put(put_config))
        .route(
            "/api/config/console_token",
            post(push_console_token).options(push_console_token_options),
        )
        .route("/pull/{task_id}/{*rest}", get(pull_serve))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state)
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn health() -> Json<Value> {
    Json(json!({"ok": true}))
}

async fn auth_middleware(
    State(relay): State<Arc<Relay>>,
    headers: HeaderMap,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    let index_path: axum::http::Uri = req.uri().clone();
    // /pull/* carries its own per-task token (aria2/curl on the instance
    // cannot send headers conveniently); everything else needs the relay token.
    let path = index_path.path();
    let is_public = path == "/"
        || path.starts_with("/pull/")
        || path == "/api/config/console_token";
    let expected = relay.config().server.api_token;
    let provided = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.strip_prefix("Bearer ").unwrap_or(v).trim().to_string());
    if !is_public && (expected.is_empty() || provided.as_deref() != Some(expected.as_str())) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "token 无效"})))
            .into_response();
    }
    next.run(req).await
}

fn err_response(e: anyhow::Error) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e:#}")}))).into_response()
}

// ---- tasks ---------------------------------------------------------------------

async fn list_tasks(State(relay): State<Arc<Relay>>) -> Json<Value> {
    Json(json!({"tasks": relay.store.list()}))
}

async fn get_task(
    State(relay): State<Arc<Relay>>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    match relay.store.get(&id) {
        Some(t) => Json(t).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "任务不存在"}))).into_response(),
    }
}

async fn create_task(
    State(relay): State<Arc<Relay>>,
    mut mp: Multipart,
) -> Response {
    let mut script_name: Option<String> = None;
    let mut script_bytes: Option<Vec<u8>> = None;
    let mut name = String::new();
    let mut output_dir = "out".to_string();
    let mut auto_upload = true;
    let mut autostart = true;
    let mut upload_mode = relay.config().autodl.upload_mode;

    while let Ok(Some(field)) = mp.next_field().await {
        match field.name().unwrap_or_default() {
            "script" => {
                script_name = field.file_name().map(|s| s.to_string());
                match field.bytes().await {
                    Ok(b) => script_bytes = Some(b.to_vec()),
                    Err(e) => return err_response(anyhow::anyhow!("读取脚本失败: {e}")),
                }
            }
            "name" => {
                name = field.text().await.unwrap_or_default();
            }
            "output_dir" => {
                output_dir = field.text().await.unwrap_or_default();
            }
            "auto_upload" => {
                auto_upload = matches!(
                    field.text().await.unwrap_or_default().as_str(),
                    "true" | "1" | "on"
                );
            }
            "autostart" => {
                autostart = matches!(
                    field.text().await.unwrap_or_default().as_str(),
                    "true" | "1" | "on" | ""
                );
            }
            "upload_mode" => {
                upload_mode = match field.text().await.unwrap_or_default().as_str() {
                    "jupyter" => UploadMode::Jupyter,
                    "sftp" => UploadMode::Sftp,
                    _ => UploadMode::Pull,
                };
            }
            _ => {}
        }
    }

    let bytes = match script_bytes {
        Some(b) if !b.is_empty() => b,
        _ => return err_response(anyhow::anyhow!("缺少脚本文件（字段名 script）")),
    };
    let script_file = script_name.unwrap_or_else(|| "run.py".into());
    let task_name = if name.trim().is_empty() {
        script_file.clone()
    } else {
        name
    };
    match relay
        .create_task(
            &task_name,
            &script_file,
            &bytes,
            &output_dir,
            auto_upload,
            upload_mode,
            autostart,
        )
        .await
    {
        Ok(task) => Json(task).into_response(),
        Err(e) => err_response(e),
    }
}

async fn task_log(
    State(relay): State<Arc<Relay>>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Response {
    let bytes: usize = q
        .get("bytes")
        .and_then(|b| b.parse().ok())
        .unwrap_or(65536)
        .min(1_048_576);
    let path = relay.store.task_dir(&id).join("task.log");
    match std::fs::read(&path) {
        Ok(content) => {
            let start = content.len().saturating_sub(bytes);
            let tail = &content[start..];
            let text = String::from_utf8_lossy(tail).to_string();
            (
                [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                text,
            )
                .into_response()
        }
        Err(_) => (
            [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            "(尚无日志)".to_string(),
        )
            .into_response(),
    }
}

async fn start_task(
    State(relay): State<Arc<Relay>>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    match Arc::clone(&relay).start_task(&id).await {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => err_response(e),
    }
}

async fn upload_task(
    State(relay): State<Arc<Relay>>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let this = Arc::clone(&relay);
    let id2 = id.clone();
    // run in background; the HTTP response returns immediately
    tokio::spawn(async move {
        if let Err(e) = Arc::clone(&this).run_pipeline(&id2).await {
            this.store.update(&id2, |t| {
                if t.status == crate::model::TaskStatus::Uploading {
                    t.status = crate::model::TaskStatus::Failed;
                    t.error = Some(format!("{e:#}"));
                    t.finished_at = Some(crate::model::now_secs());
                }
            })
            .ok();
        }
    });
    Json(json!({"ok": true, "message": "上传流程已在后台启动，请通过日志/状态轮询"})).into_response()
}

async fn cancel_task(
    State(relay): State<Arc<Relay>>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    match relay.cancel(&id).await {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => err_response(e),
    }
}

async fn delete_task(
    State(relay): State<Arc<Relay>>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    match relay.store.remove(&id) {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => err_response(e),
    }
}

// ---- autodl --------------------------------------------------------------------

async fn list_instances(State(relay): State<Arc<Relay>>) -> Response {
    match relay.autodl_instances().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn autodl_status(State(relay): State<Arc<Relay>>) -> Response {
    match relay.autodl_status().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn boot_no_gpu(State(relay): State<Arc<Relay>>) -> Response {
    match relay.autodl_boot_no_gpu().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn power_off(State(relay): State<Arc<Relay>>) -> Response {
    match relay.autodl_power_off().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn autodl_disk(State(relay): State<Arc<Relay>>) -> Response {
    match relay.autodl_disk().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn list_remote(State(relay): State<Arc<Relay>>) -> Response {
    match relay.list_remote().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err_response(e),
    }
}

async fn delete_remote(
    State(relay): State<Arc<Relay>>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(path) = q.get("path").cloned() else {
        return err_response(anyhow::anyhow!("缺少 path 参数"));
    };
    match relay.delete_remote(&path).await {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => err_response(e),
    }
}

// ---- AutoDL 账密登录（WebUI 表单 → 官方 new_login，换取 console_token）----

/// Depth-limited search for the first JWT-looking string in a JSON response
/// (field naming is not documented, so match the shape: eyJ… with two dots).
fn find_jwt(v: &Value) -> Option<String> {
    fn walk(v: &Value, depth: u8) -> Option<String> {
        if depth > 5 {
            return None;
        }
        match v {
            Value::String(s) => {
                let s = s.trim();
                (s.starts_with("eyJ") && s.matches('.').count() == 2 && s.len() > 60)
                    .then(|| s.to_string())
            }
            Value::Object(m) => m.values().find_map(|x| walk(x, depth + 1)),
            Value::Array(a) => a.iter().find_map(|x| walk(x, depth + 1)),
            _ => None,
        }
    }
    walk(v, 0)
}

/// Forward credentials to AutoDL's own login API once, keep the returned JWT
/// as console_token, and switch to console mode (regular instances + 无卡开机).
/// The password is never stored or logged.
/// 保存 token 并切换 console 模式（两条成功路径共用）。
fn finish_login(relay: &Arc<Relay>, token: &str) -> Response {
    let masked = format!(
        "{}…（{} 字符）",
        &token[..token.len().min(8)],
        token.len()
    );
    let mut c = relay.config();
    c.apply_patch(&json!({"console_token": token, "api_mode": "console"}));
    if let Err(e) = relay.persist_config(&c) {
        return err_response(e);
    }
    tracing::info!("AutoDL 账密登录成功，console_token 已更新（{}），api_mode -> console", masked);
    Json(json!({
        "ok": true,
        "message": "登录成功：console_token 已保存，api_mode 已切换为 console（立即生效）",
        "console_token": masked,
    }))
    .into_response()
}

/// 用 new_login / 扫码返回的 ticket 换取真正的 JWT（/api/v1/passport）。
async fn ticket_to_token(ticket: &str) -> anyhow::Result<String> {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()?;
    let resp = http
        .post("https://www.autodl.com/api/v1/passport")
        .json(&json!({"ticket": ticket}))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("passport 请求失败: {e}"))?;
    let v: Value = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("passport 返回非 JSON: {e}"))?;
    let code = v.get("code").and_then(|c| c.as_str()).unwrap_or("");
    if code != "Success" {
        let msg = v.get("msg").and_then(|m| m.as_str()).unwrap_or("未知错误");
        anyhow::bail!("ticket 换取 token 失败: {msg}");
    }
    find_jwt(&v).ok_or_else(|| anyhow::anyhow!("passport 响应中未识别到 token"))
}

/// 微信扫码登录第一步：申请二维码会话（uuid + 二维码图片 URL）。
async fn wx_login_start(State(relay): State<Arc<Relay>>) -> Response {
    let _ = &relay;
    let http = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
    {
        Ok(c) => c,
        Err(e) => return err_response(anyhow::anyhow!("{e}")),
    };
    let resp = http
        .get("https://www.autodl.com/api/v1/wx/login")
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => return err_response(anyhow::anyhow!("获取微信二维码失败: {e}")),
    };
    let v: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return err_response(anyhow::anyhow!("非 JSON 响应: {e}")),
    };
    let code = v.get("code").and_then(|c| c.as_str()).unwrap_or("");
    if code != "Success" {
        let msg = v.get("msg").and_then(|m| m.as_str()).unwrap_or("未知错误");
        return err_response(anyhow::anyhow!("获取微信二维码失败: {msg}"));
    }
    Json(v.get("data").cloned().unwrap_or(json!(null))).into_response()
}

/// 微信扫码登录第二步：轮询扫码状态；确认后 ticket → passport → JWT → 保存。
async fn wx_login_poll(
    State(relay): State<Arc<Relay>>,
    Json(body): Json<Value>,
) -> Response {
    let uuid = body
        .get("uuid")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if uuid.is_empty() {
        return err_response(anyhow::anyhow!("缺少 uuid"));
    }
    let http = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
    {
        Ok(c) => c,
        Err(e) => return err_response(anyhow::anyhow!("{e}")),
    };
    let resp = http
        .post("https://www.autodl.com/api/v1/wx/polling/query")
        .json(&json!({"uuid": uuid}))
        .send()
        .await;
    let resp = match resp {
        Ok(r) => r,
        Err(e) => return err_response(anyhow::anyhow!("轮询请求失败: {e}")),
    };
    let v: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return err_response(anyhow::anyhow!("非 JSON 响应: {e}")),
    };
    let code = v.get("code").and_then(|c| c.as_str()).unwrap_or("");
    if code != "Success" {
        let msg = v.get("msg").and_then(|m| m.as_str()).unwrap_or("未知错误");
        return Json(json!({"status": "error", "message": msg})).into_response();
    }
    let data = v.get("data").cloned().unwrap_or(Value::Null);
    if data.is_null() {
        return Json(json!({"status": "waiting"})).into_response();
    }
    let ticket = data
        .get("ticket")
        .and_then(|t| t.as_str())
        .map(|s| s.to_string());
    match ticket {
        Some(t) => match ticket_to_token(&t).await {
            Ok(token) => finish_login(&relay, &token),
            Err(e) => err_response(e),
        },
        None => Json(json!({"status": "scanned", "message": "已扫码，请在微信上确认"}))
            .into_response(),
    }
}

// ---- console_token push (用户脚本/书签从 autodl.com 推送，免 CORS 预检) ----

/// Accept a freshly grabbed console_token pushed from a userscript or
/// bookmarklet running on https://www.autodl.com. Auth via `?t=<api_token>`
/// (a cross-origin fetch cannot send Authorization headers without a
/// preflight, so the token goes in the URL and the body stays text/plain).
async fn push_console_token(
    State(relay): State<Arc<Relay>>,
    Query(q): Query<std::collections::HashMap<String, String>>,
    body: String,
) -> Response {
    let cfg = relay.config();
    if q.get("t").map(|t| t.as_str() == cfg.server.api_token) != Some(true) {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "bad push token"}))).into_response();
    }
    let v: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return err_response(anyhow::anyhow!("body 需为 JSON: {e}")),
    };
    let Some(tok) = v.get("console_token").and_then(|x| x.as_str()) else {
        return err_response(anyhow::anyhow!("缺少 console_token 字段"));
    };
    if tok.len() < 20 {
        return err_response(anyhow::anyhow!("token 长度异常（{} 字符），请整串复制", tok.len()));
    }
    let mut c = relay.config();
    c.apply_patch(&json!({"console_token": tok}));
    if let Err(e) = relay.persist_config(&c) {
        return err_response(e);
    }
    tracing::info!("console_token 已通过用户脚本推送更新（{} 字符）", tok.len());
    let mut resp = Json(json!({"ok": true, "message": "console_token 已更新并持久化"})).into_response();
    resp.headers_mut().insert(
        "Access-Control-Allow-Origin",
        HeaderValue::from_static("https://www.autodl.com"),
    );
    resp
}

/// Defensive preflight answer (text/plain bodies skip preflight, but be
/// permissive anyway so any header combination still works).
async fn push_console_token_options() -> Response {
    let mut resp = StatusCode::NO_CONTENT.into_response();
    let h = resp.headers_mut();
    h.insert("Access-Control-Allow-Origin", HeaderValue::from_static("https://www.autodl.com"));
    h.insert("Access-Control-Allow-Methods", HeaderValue::from_static("POST, OPTIONS"));
    h.insert("Access-Control-Allow-Headers", HeaderValue::from_static("Content-Type"));
    resp
}

// ---- pull channel (高速通道) -----------------------------------------------------

/// Serve a task's output dir over HTTP with Range support so the no-GPU
/// instance can pull data with aria2c (16 connections, resumable). Auth is the
/// per-task `?t=<pull_token>` issued when the pull pipeline starts.
async fn pull_serve(
    State(relay): State<Arc<Relay>>,
    AxumPath((task_id, _rest)): AxumPath<(String, String)>,
    Query(q): Query<std::collections::HashMap<String, String>>,
    req: axum::extract::Request,
) -> Response {
    let Some(task) = relay.store.get(&task_id) else {
        return (StatusCode::NOT_FOUND, "task not found").into_response();
    };
    let token_ok = task
        .pull_token
        .as_deref()
        .is_some_and(|t| Some(t) == q.get("t").map(|s| s.as_str()));
    if !token_ok {
        return (StatusCode::FORBIDDEN, "bad pull token").into_response();
    }
    let Some(root) = relay.pull_root(&task_id) else {
        return (StatusCode::NOT_FOUND, "output dir missing").into_response();
    };
    // strip the /pull/<task_id> prefix; ServeDir resolves from the output root
    let (mut parts, body) = req.into_parts();
    let suffix = parts.uri.path().splitn(4, '/').nth(3).unwrap_or("");
    if let Ok(uri) = axum::http::Uri::builder()
        .path_and_query(format!("/{suffix}"))
        .build()
    {
        parts.uri = uri;
    }
    let req = axum::extract::Request::from_parts(parts, body);
    use tower::util::ServiceExt as _;
    match tower_http::services::ServeDir::new(root).oneshot(req).await {
        Ok(resp) => resp.map(axum::body::Body::new),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("serve error: {e}"),
        )
            .into_response(),
    }
}

// ---- config --------------------------------------------------------------------

async fn get_config(State(relay): State<Arc<Relay>>) -> Json<Value> {
    let cfg = relay.config();
    let mask = |s: &str| {
        if s.is_empty() {
            json!("")
        } else {
            json!(format!("{}***", &s[..s.char_indices().nth(4).map(|(i, _)| i).unwrap_or(s.len())]))
        }
    };
    Json(json!({
        "api_mode": cfg.autodl.api_mode,
        "console_token": mask(&cfg.autodl.console_token),
        "console_token_updated_at": cfg.autodl.console_token_updated_at,
        "dev_token": mask(&cfg.autodl.dev_token),
        "instance_uuid": cfg.autodl.instance_uuid,
        "ssh_password_set": !cfg.autodl.ssh_password.is_empty(),
        "upload_mode": cfg.autodl.upload_mode.as_str(),
        "remote_root": cfg.autodl.remote_root,
        "boot_no_gpu": cfg.autodl.boot_no_gpu,
        "power_off_after_upload": cfg.autodl.power_off_after_upload,
        "allow_gpu_boot": cfg.autodl.allow_gpu_boot,
        "strict_host_key": cfg.autodl.strict_host_key,
        "disk": {
            "max_used_ratio": cfg.disk.max_used_ratio,
            "reserved_gb": cfg.disk.reserved_gb,
            "auto_clean": cfg.disk.auto_clean,
            "keep_last_uploads": cfg.disk.keep_last_uploads,
        },
        "runner_python": cfg.runner.python,
    }))
}

async fn put_config(
    State(relay): State<Arc<Relay>>,
    Json(patch): Json<Value>,
) -> Response {
    let mut cfg = relay.config();
    cfg.apply_patch(&patch);
    match relay.persist_config(&cfg) {
        Ok(()) => get_config(State(relay)).await.into_response(),
        Err(e) => err_response(e),
    }
}
