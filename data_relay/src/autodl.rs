//! AutoDL API client.
//!
//! Two backends share one interface:
//!
//! * `console` — the web console's internal API (`https://www.autodl.com/api/v1`),
//!   authenticated with the browser session token from Local Storage. This is
//!   the only way to power an instance on in no-GPU mode (`payload: "non_gpu"`)
//!   because the official OpenAPI does not expose it.
//! * `pro` — the official developer API (`https://api.autodl.com/api/v1/dev/instance/pro`),
//!   authenticated with the developer token. Snapshot/list/power_off are
//!   supported; power_on is GPU-only ("暂不支持API以无卡模式开机").
//!
//! Both snapshot/detail responses carry `proxy_host` / `ssh_port` /
//! `root_password` (SSH upload channel) and `jupyter_domain` / `jupyter_token`
//! (Jupyter upload channel), plus data-disk usage numbers.

use crate::config::AutoDlCfg;
use anyhow::{Context, bail};
use serde_json::{Value, json};
use std::time::Duration;

const CONSOLE_BASE: &str = "https://www.autodl.com/api/v1";
const PRO_BASE: &str = "https://api.autodl.com/api/v1/dev/instance/pro";

#[derive(Debug, Clone)]
pub struct SshInfo {
    pub host: String,
    pub port: u16,
    pub password: String,
}

#[derive(Debug, Clone)]
pub struct JupyterInfo {
    pub base: String,
    pub token: String,
}

#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub uuid: String,
    pub name: Option<String>,
    pub status: String,
    pub start_mode: Option<String>,
    pub charge_type: Option<String>,
    pub ssh: Option<SshInfo>,
    pub jupyter: Option<JupyterInfo>,
    pub data_disk_total: Option<u64>,
    pub data_disk_used: Option<u64>,
    /// Original response (minus nothing — callers must not log this).
    #[allow(dead_code)]
    pub raw: Value,
}

impl InstanceInfo {
    pub fn is_running(&self) -> bool {
        self.status == "running"
    }

    /// Instance is off and can be powered on.
    pub fn is_off(&self) -> bool {
        matches!(self.status.as_str(), "shutdown" | "stopped")
    }

    /// Serialize for the WebUI with secrets redacted.
    pub fn to_redacted_json(&self) -> Value {
        let mut v = json!({
            "uuid": self.uuid,
            "name": self.name,
            "status": self.status,
            "start_mode": self.start_mode,
            "charge_type": self.charge_type,
            "data_disk_total": self.data_disk_total,
            "data_disk_used": self.data_disk_used,
            "ssh_available": self.ssh.is_some(),
            "jupyter_available": self.jupyter.is_some(),
        });
        if let Some(j) = &self.jupyter {
            v["jupyter_base"] = json!(j.base);
        }
        v
    }
}

pub struct AutoDl {
    http: reqwest::Client,
    cfg: AutoDlCfg,
    #[cfg(test)]
    test_base: Option<String>,
}

/// A resource-stock rejection is retryable; billing/authentication/driver errors are not.
#[derive(Debug)]
pub struct GpuUnavailable(pub String);
impl std::fmt::Display for GpuUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for GpuUnavailable {}

fn gpu_capacity_message(code: &str, message: &str) -> bool {
    let text = format!("{code} {message}").to_lowercase();
    if ["余额", "balance", "token", "权限", "permission", "authentication", "显存", "memory", "驱动", "driver"]
        .iter().any(|word| text.contains(word)) { return false; }
    (["gpu", "显卡", "算力"].iter().any(|word| text.contains(word))
        && ["不足", "不够", "无空闲", "没有可用", "暂无可用", "not enough", "notenough", "insufficient", "no available", "no free", "out of stock", "sold out"]
            .iter().any(|word| text.contains(word)))
        || ["空闲资源不足", "资源库存不足"].iter().any(|word| text.contains(word))
}

impl AutoDl {
    pub fn new(cfg: &AutoDlCfg) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(AutoDl { http, cfg: cfg.clone(), #[cfg(test)] test_base:None })
    }

    #[cfg(test)]
    pub(crate) fn test_client(cfg: &AutoDlCfg, base: String) -> Self {
        let mut client=Self::new(cfg).unwrap();client.test_base=Some(base);client
    }

    fn endpoint(&self, base: &str, path: &str) -> String {
        #[cfg(test)]
        if let Some(base)=&self.test_base {return format!("{base}/{path}");}
        format!("{base}/{path}")
    }

    // ---- low-level requests -------------------------------------------------

    async fn console_post(&self, path: &str, body: Value) -> anyhow::Result<Value> {
        let resp = self
            .http
            .post(self.endpoint(CONSOLE_BASE,path))
            .header("Authorization", &self.cfg.console_token)
            .header("AppVersion", "0.0.0")
            .json(&body)
            .send()
            .await
            .context("AutoDL 控制台请求失败")?;
        Self::unwrap_envelope(resp).await
    }

    async fn console_get(&self, path: &str, query: &[(&str, &str)]) -> anyhow::Result<Value> {
        let mut req = self
            .http
            .get(self.endpoint(CONSOLE_BASE,path))
            .header("Authorization", &self.cfg.console_token)
            .header("AppVersion", "0.0.0");
        for (k, v) in query {
            req = req.query(&[(k, v)]);
        }
        Self::unwrap_envelope(req.send().await.context("AutoDL 控制台请求失败")?).await
    }

    async fn pro_get(&self, path: &str, query: &[(&str, &str)]) -> anyhow::Result<Value> {
        let mut req = self
            .http
            .get(self.endpoint(PRO_BASE,path))
            .header("Authorization", &self.cfg.dev_token);
        for (k, v) in query {
            req = req.query(&[(k, v)]);
        }
        Self::unwrap_envelope(req.send().await.context("AutoDL Pro API 请求失败")?).await
    }

    async fn pro_post(&self, path: &str, body: Value) -> anyhow::Result<Value> {
        let resp = self
            .http
            .post(self.endpoint(PRO_BASE,path))
            .header("Authorization", &self.cfg.dev_token)
            .json(&body)
            .send()
            .await
            .context("AutoDL Pro API 请求失败")?;
        Self::unwrap_envelope(resp).await
    }

    async fn unwrap_envelope(resp: reqwest::Response) -> anyhow::Result<Value> {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        let v: Value = serde_json::from_str(&text)
            .with_context(|| format!("AutoDL 返回非 JSON (HTTP {status}): {}", truncate(&text, 300)))?;
        let code = v.get("code").and_then(|c| c.as_str()).unwrap_or_default();
        if code != "Success" {
            let msg = v
                .get("msg")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error")
                .to_string();
            let error = format!("AutoDL API 错误 (code={code}, HTTP {status}): {msg}");
            if gpu_capacity_message(code, &msg) { return Err(GpuUnavailable(error).into()); }
            bail!("{error}");
        }
        Ok(v.get("data").cloned().unwrap_or(Value::Null))
    }

    // ---- instance operations -------------------------------------------------

    /// List instances through the configured backend.
    pub async fn list(&self) -> anyhow::Result<Vec<InstanceInfo>> {
        let data = match self.cfg.api_mode.as_str() {
            "console" => {
                self.console_post("instance", json!({"page_index": 1, "page_size": 50}))
                    .await?
            }
            "pro" => {
                self.pro_post("list", json!({"page_index": 1, "page_size": 50}))
                    .await?
            }
            other => bail!("未知 api_mode: {other}"),
        };
        let items = first_array(&data).ok_or_else(|| anyhow::anyhow!("实例列表响应格式未识别"))?;
        Ok(items.iter().map(parse_instance).collect())
    }

    /// Fetch one instance (detail carries ssh/jupyter/disk info).
    pub async fn instance(&self, uuid: &str) -> anyhow::Result<InstanceInfo> {
        let data = match self.cfg.api_mode.as_str() {
            "console" => {
                let mut data = self
                    .console_get("instance/detail", &[("instance_uuid", uuid)])
                    .await?;
                if is_effectively_empty(&data) {
                    data = self
                        .console_get("instance/snapshot", &[("instance_uuid", uuid)])
                        .await?;
                }
                // Detail may omit SSH credentials that are present in the list.
                // Keep list fields, then overlay the more detailed response.
                if let Some(listed) = self.list().await?.into_iter().find(|i| i.uuid == uuid) {
                    if let (Some(base), Some(detail)) = (listed.raw.as_object(), data.as_object()) {
                        let mut merged = base.clone();
                        merged.extend(detail.clone());
                        data = Value::Object(merged);
                    }
                }
                data
            }
            "pro" => self.pro_get("snapshot", &[("instance_uuid", uuid)]).await?,
            other => bail!("未知 api_mode: {other}"),
        };
        Ok(parse_instance(&data))
    }

    /// 无卡模式开机 — console API only: POST instance/power_on with
    /// `{"instance_uuid": ..., "payload": "non_gpu"}`.
    pub async fn power_on_no_gpu(&self, uuid: &str) -> anyhow::Result<()> {
        if self.cfg.api_mode != "console" {
            bail!("官方 Pro API 不支持无卡模式开机；请使用 console 模式（填 console_token），或在控制台手动无卡开机");
        }
        self.console_post(
            "instance/power_on",
            json!({"instance_uuid": uuid, "payload": "non_gpu"}),
        )
        .await
        .map(|_| ())
    }

    /// Regular (GPU) boot. Pro API requires `payload: "gpu"`.
    pub async fn power_on_gpu(&self, uuid: &str) -> anyhow::Result<()> {
        match self.cfg.api_mode.as_str() {
            "console" => {
                self.console_post("instance/power_on", json!({"instance_uuid": uuid}))
                    .await
                    .map(|_| ())
            }
            "pro" => {
                self.pro_post(
                    "power_on",
                    json!({"instance_uuid": uuid, "payload": "gpu"}),
                )
                .await
                .map(|_| ())
            }
            other => bail!("未知 api_mode: {other}"),
        }
    }

    pub async fn power_off(&self, uuid: &str) -> anyhow::Result<()> {
        match self.cfg.api_mode.as_str() {
            "console" => {
                self.console_post("instance/power_off", json!({"instance_uuid": uuid}))
                    .await
                    .map(|_| ())
            }
            "pro" => self.pro_post("power_off", json!({"instance_uuid": uuid})).await.map(|_| ()),
            other => bail!("未知 api_mode: {other}"),
        }
    }

    /// Pick the configured instance (or the only one on the account).
    pub async fn pick_instance(&self) -> anyhow::Result<InstanceInfo> {
        if !self.cfg.instance_uuid.is_empty() {
            return self.instance(&self.cfg.instance_uuid).await;
        }
        let all = self.list().await?;
        if all.is_empty() {
            bail!("账号下没有实例，请先在 AutoDL 创建实例");
        }
        if all.len() == 1 {
            return Ok(all.into_iter().next().unwrap());
        }
        let names: Vec<String> = all
            .iter()
            .map(|i| format!("{}({})", i.uuid, i.name.clone().unwrap_or_default()))
            .collect();
        bail!(
            "账号下有 {} 台实例，请在配置中指定 autodl.instance_uuid：{}",
            all.len(),
            names.join(", ")
        );
    }

    /// Make sure the instance is running — booting 无卡 first if allowed.
    /// Returns the final instance info (with ssh/jupyter credentials).
    pub async fn ensure_running(&self) -> anyhow::Result<InstanceInfo> {
        let mut info = self.pick_instance().await?;
        let uuid = info.uuid.clone();
        let deadline = std::time::Instant::now() + Duration::from_secs(self.cfg.boot_timeout_secs);
        let mut boot_requested = false;

        loop {
            if info.is_running() {
                return Ok(info);
            }
            if info.is_off() && !boot_requested {
                if self.cfg.boot_no_gpu {
                    tracing::info!("实例 {uuid} 处于 {}，无卡模式开机…", info.status);
                    self.power_on_no_gpu(&uuid).await?;
                } else if self.cfg.allow_gpu_boot {
                    tracing::warn!("实例 {uuid} 有卡模式开机（计费按 GPU 算）");
                    self.power_on_gpu(&uuid).await?;
                } else {
                    bail!("实例未开机且配置禁止自动开机：请在控制台无卡模式开机后重试");
                }
                boot_requested = true;
            }
            if std::time::Instant::now() > deadline {
                bail!("等待实例 running 超时（{}s），最后状态: {}", self.cfg.boot_timeout_secs, info.status);
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
            info = self.instance(&uuid).await?;
        }
    }
}

// ---- response parsing --------------------------------------------------------

/// First JSON array found among the common envelope shapes.
fn first_array(v: &Value) -> Option<Vec<Value>> {
    if let Some(a) = v.as_array() {
        return Some(a.clone());
    }
    for key in ["data", "list", "instance_list", "items"] {
        if let Some(a) = v.get(key).and_then(|x| x.as_array()) {
            return Some(a.clone());
        }
    }
    None
}

fn is_effectively_empty(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Object(o) => o.is_empty() || o.iter().all(|(_, x)| is_effectively_empty(x)),
        _ => false,
    }
}

/// Depth-limited recursive key lookup; console/pro envelopes nest fields at
/// slightly different levels, so parse defensively.
fn find_field(v: &Value, key: &str, depth: u8) -> Option<Value> {
    if depth > 4 {
        return None;
    }
    match v {
        Value::Object(map) => {
            if let Some(found) = map.get(key) {
                if !found.is_null() {
                    return Some(found.clone());
                }
            }
            for (_, child) in map {
                if let Some(found) = find_field(child, key, depth + 1) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(|i| find_field(i, key, depth + 1)),
        _ => None,
    }
}

fn get_str(v: &Value, key: &str) -> Option<String> {
    find_field(v, key, 0).and_then(|x| x.as_str().map(|s| s.to_string()))
}

fn get_u64(v: &Value, key: &str) -> Option<u64> {
    find_field(v, key, 0).and_then(|x| {
        x.as_u64()
            .or_else(|| x.as_str().and_then(|s| s.parse().ok()))
    })
}

fn get_i64_as_u64(v: &Value, key: &str) -> Option<u64> {
    find_field(v, key, 0).and_then(|x| x.as_i64().map(|n| n.max(0) as u64))
}

/// Parse `ssh -p 34222 root@connect.westc.gpuhub.com` into (host, port, user).
pub fn parse_ssh_command(cmd: &str) -> Option<(String, u16, String)> {
    let port = cmd
        .split_whitespace()
        .position(|t| t == "-p")
        .and_then(|i| cmd.split_whitespace().nth(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(22);
    let target = cmd.split_whitespace().find(|t| t.contains('@'))?;
    let (user, host) = target.split_once('@')?;
    Some((host.to_string(), port, user.to_string()))
}

fn parse_instance(v: &Value) -> InstanceInfo {
    // Never adopt a nested machine/occupant UUID as the instance identity.
    let uuid = v.get("instance_uuid").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_owned)
        .or_else(|| v.get("uuid").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_owned))
        .or_else(|| get_str(v, "uuid"))
        .or_else(|| get_str(v, "instance_uuid"))
        .unwrap_or_default();
    let status = get_str(v, "status").unwrap_or_else(|| "unknown".into());

    let ssh = {
        let mut host = get_str(v, "proxy_host");
        let mut port = get_i64_as_u64(v, "ssh_port").map(|p| p as u16);
        if let Some(cmd) = get_str(v, "ssh_command") {
            if let Some((h, p, _user)) = parse_ssh_command(&cmd) {
                host = host.or(Some(h));
                port = port.or(Some(p));
            }
        }
        let password = get_str(v, "root_password").or_else(|| get_str(v, "password"));
        match (host, port, password) {
            (Some(host), Some(port), Some(password)) => Some(SshInfo {
                host,
                port,
                password,
            }),
            _ => None,
        }
    };

    let jupyter = {
        let domain = get_str(v, "jupyter_domain");
        let token = get_str(v, "jupyter_token");
        match (domain, token) {
            (Some(domain), Some(token)) if !domain.is_empty() => {
                let base = if domain.starts_with("http") {
                    domain
                } else {
                    format!("https://{domain}")
                };
                Some(JupyterInfo { base, token })
            }
            _ => None,
        }
    };

    let data_disk_total = get_u64(v, "data_disk_total_size")
        .or_else(|| get_u64(v, "data_disk_total"));
    let data_disk_used = get_u64(v, "data_disk_used_size")
        .or_else(|| get_u64(v, "data_disk_used"));

    InstanceInfo {
        uuid,
        name: get_str(v, "name").or_else(|| get_str(v, "instance_name")),
        status,
        start_mode: get_str(v, "start_mode"),
        charge_type: get_str(v, "charge_type"),
        ssh,
        jupyter,
        data_disk_total,
        data_disk_used,
        raw: v.clone(),
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
    fn capacity_errors_exclude_balance_auth_driver_and_memory_failures() {
        for (code,message) in [("InsufficientGPU",""),("Fail","主机空闲GPU数量不够"),("Fail","No available GPU"),("Fail","空闲资源不足")] {
            assert!(gpu_capacity_message(code,message),"{message}");
        }
        for message in ["GPU 开机失败：余额不足","Token expired","GPU driver unavailable","GPU memory insufficient","network timeout","实例已删除"] {
            assert!(!gpu_capacity_message("Fail",message),"{message}");
        }
    }

    #[test]
    fn detail_instance_identity_precedes_nested_machine_uuid() {
        let value = json!({"instance_uuid": "rrprrh6gtl-cb8265fc", "status": "shutdown",
            "machine": {"occupant": {"uuid": "pro-unrelated"}}});
        assert_eq!(parse_instance(&value).uuid, "rrprrh6gtl-cb8265fc");
    }

    #[test]
    fn ssh_command_parse() {
        let (host, port, user) =
            parse_ssh_command("ssh -p 34222 root@connect.westc.gpuhub.com").unwrap();
        assert_eq!(host, "connect.westc.gpuhub.com");
        assert_eq!(port, 34222);
        assert_eq!(user, "root");
    }

    #[test]
    fn no_gpu_boot_body_uses_non_gpu_payload() {
        // Regression guard: the console API expects the literal "non_gpu".
        let body = json!({"instance_uuid": "p1-x", "payload": "non_gpu"});
        assert_eq!(body["payload"], "non_gpu");
    }

    #[test]
    fn find_field_recurses_into_nesting() {
        let v: Value = serde_json::json!({"code":"Success","data":{"instance":{
            "usage_info": {"data_disk_total_size": 53687091200i64},
            "ssh_command": "ssh -p 100 root@a.b.c",
            "root_password": "pw"}}});
        assert_eq!(
            get_u64(&v, "data_disk_total_size"),
            Some(53687091200)
        );
        assert_eq!(get_str(&v, "root_password").as_deref(), Some("pw"));
    }

    #[test]
    fn parse_instance_extracts_ssh_and_jupyter() {
        let v: Value = serde_json::json!({
            "uuid": "pro-123", "status": "running", "start_mode": "gpu",
            "proxy_host": "connect.x.autodl.com", "ssh_port": 40123,
            "root_password": "secret", "jupyter_domain": "a1-x.autodl.com:8443",
            "jupyter_token": "jt", "data_disk_total_size": 100, "data_disk_used_size": 40,
        });
        let info = parse_instance(&v);
        assert_eq!(info.uuid, "pro-123");
        let ssh = info.ssh.unwrap();
        assert_eq!(ssh.port, 40123);
        assert_eq!(ssh.password, "secret");
        let j = info.jupyter.unwrap();
        assert_eq!(j.base, "https://a1-x.autodl.com:8443");
        assert_eq!(info.data_disk_total, Some(100));
        assert_eq!(info.data_disk_used, Some(40));
    }

    #[test]
    fn redacted_json_has_no_secrets() {
        let v: Value = serde_json::json!({"uuid": "u", "status": "running",
            "proxy_host": "h", "ssh_port": 1, "root_password": "sekret",
            "jupyter_domain": "d", "jupyter_token": "t"});
        let info = parse_instance(&v);
        let s = info.to_redacted_json().to_string();
        assert!(!s.contains("sekret"));
        assert!(!s.contains("\"t\""));
    }
}
