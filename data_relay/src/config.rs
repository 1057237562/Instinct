//! TOML configuration with defaults for every field.

use crate::model::UploadMode;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct Config {
    pub server: ServerCfg,
    pub runner: RunnerCfg,
    pub autodl: AutoDlCfg,
    pub disk: DiskCfg,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerCfg {
    /// Listen address of the relay WebUI/API.
    pub listen: String,
    /// Bearer token required by every /api call.
    pub api_token: String,
    /// Working directory for task workspaces and state.
    pub data_dir: String,
}

impl Default for ServerCfg {
    fn default() -> Self {
        ServerCfg {
            listen: "0.0.0.0:7800".into(),
            api_token: "change-me".into(),
            data_dir: "relay_data".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RunnerCfg {
    /// Interpreter for uploaded .py scripts.
    pub python: String,
    /// Extra environment injected into every script run.
    pub env: BTreeMap<String, String>,
}

impl Default for RunnerCfg {
    fn default() -> Self {
        RunnerCfg {
            python: "python3".into(),
            env: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoDlCfg {
    /// "console" — www.autodl.com console API (supports 无卡开机 with the
    ///   browser session token); "pro" — official api.autodl.com dev API.
    pub api_mode: String,
    /// Token from www.autodl.com console: browser DevTools -> Application ->
    /// Local Storage -> `token` (used as `Authorization` header verbatim).
    pub console_token: String,
    /// Developer token from console -> 设置 -> 开发者Token (pro mode only).
    pub dev_token: String,
    /// Instance uuid ("p1-...", "pro-..."). Empty = auto-pick when the
    /// account has exactly one instance.
    pub instance_uuid: String,
    /// Optional SSH password override; by default the per-instance
    /// `root_password` from the instance snapshot/detail is used.
    pub ssh_password: String,
    pub ssh_user: String,
    /// Default upload channel for new tasks (per-task override possible).
    pub upload_mode: UploadMode,
    /// Managed upload root on the instance data disk. Everything the relay
    /// creates and may delete lives under this directory.
    pub remote_root: String,
    /// Boot in no-GPU mode (无卡模式) when the instance is off.
    pub boot_no_gpu: bool,
    /// Power the instance off after a successful upload (无卡计费 ~0.1元/时).
    pub power_off_after_upload: bool,
    /// Pro API can only boot with GPU; guard it behind an explicit flag.
    pub allow_gpu_boot: bool,
    pub boot_timeout_secs: u64,
    /// When true, the SSH host key fingerprint must match the recorded one.
    pub strict_host_key: bool,
    /// Publicly reachable base URL of this relay for the pull channel
    /// (高速通道): e.g. `http://<relay-ip>:7800`. The no-GPU instance pulls
    /// task data from here with aria2c — bulk data never traverses SSH.
    pub pull_base_url: String,
    /// Last time console_token was set/rotated (unix secs), shown in the WebUI.
    pub console_token_updated_at: Option<i64>,
}

impl Default for AutoDlCfg {
    fn default() -> Self {
        AutoDlCfg {
            api_mode: "console".into(),
            console_token: String::new(),
            dev_token: String::new(),
            instance_uuid: String::new(),
            ssh_password: String::new(),
            ssh_user: "root".into(),
            upload_mode: UploadMode::Pull,
            remote_root: "/root/autodl-tmp/relay_uploads".into(),
            boot_no_gpu: true,
            power_off_after_upload: true,
            allow_gpu_boot: false,
            boot_timeout_secs: 900,
            strict_host_key: false,
            pull_base_url: String::new(),
            console_token_updated_at: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DiskCfg {
    /// Refuse uploads that would push data-disk usage above this fraction.
    pub max_used_ratio: f64,
    /// Keep this much headroom (GiB) below the max_used_ratio line.
    pub reserved_gb: f64,
    /// Delete oldest managed upload dirs to make room when pre-check fails.
    pub auto_clean: bool,
    /// Never delete the newest N managed upload dirs.
    pub keep_last_uploads: u32,
    /// Files above this size are rejected in jupyter upload mode.
    pub jupyter_max_file_mb: u64,
}

impl Default for DiskCfg {
    fn default() -> Self {
        DiskCfg {
            max_used_ratio: 0.92,
            reserved_gb: 2.0,
            auto_clean: true,
            keep_last_uploads: 3,
            jupyter_max_file_mb: 64,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Config> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("读取配置 {} 失败: {e}", path.display()))?;
        let cfg: Config = toml::from_str(&text)?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let text = toml::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.server.api_token.is_empty() || self.server.api_token == "change-me" {
            tracing::warn!("server.api_token 仍为默认值，请尽快修改");
        }
        match self.autodl.api_mode.as_str() {
            "console" => {
                if self.autodl.console_token.is_empty() {
                    tracing::warn!("autodl.console_token 为空：将无法自动无卡开机/关机，只能查询与上传");
                }
            }
            "pro" => {
                if self.autodl.dev_token.is_empty() {
                    anyhow::bail!("autodl.api_mode=pro 需要填写 autodl.dev_token（控制台->设置->开发者Token）");
                }
            }
            other => anyhow::bail!("autodl.api_mode 只支持 console / pro，当前为 {other}"),
        }
        if self.disk.max_used_ratio <= 0.0 || self.disk.max_used_ratio > 1.0 {
            anyhow::bail!("disk.max_used_ratio 必须在 (0, 1] 区间");
        }
        Ok(())
    }

    /// Overwrite secrets/instance selection and persist to disk. Only the
    /// listed fields are touched so the rest of the file is preserved.
    pub fn apply_patch(&mut self, patch: &serde_json::Value) {
        let autodl = &mut self.autodl;
        let get = |k: &str| patch.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
        if let Some(v) = get("api_mode") {
            if v == "console" || v == "pro" {
                autodl.api_mode = v;
            }
        }
        if let Some(v) = get("console_token") {
            if v != autodl.console_token {
                autodl.console_token_updated_at = Some(crate::model::now_secs());
            }
            autodl.console_token = v;
        }
        if let Some(v) = get("dev_token") {
            autodl.dev_token = v;
        }
        if let Some(v) = get("instance_uuid") {
            autodl.instance_uuid = v;
        }
        if let Some(v) = get("ssh_password") {
            autodl.ssh_password = v;
        }
        if let Some(v) = patch.get("power_off_after_upload").and_then(|v| v.as_bool()) {
            autodl.power_off_after_upload = v;
        }
        if let Some(v) = patch.get("boot_no_gpu").and_then(|v| v.as_bool()) {
            autodl.boot_no_gpu = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_example_config() {
        let text = include_str!("../config.example.toml");
        let cfg: Config = toml::from_str(text).expect("example config must parse");
        assert_eq!(cfg.autodl.api_mode, "console");
        assert_eq!(cfg.autodl.remote_root, "/root/autodl-tmp/relay_uploads");
        assert!(cfg.disk.max_used_ratio > 0.0);
        assert_eq!(cfg.autodl.upload_mode, UploadMode::Pull);
    }

    #[test]
    fn defaults_fill_missing_fields() {
        let cfg: Config = toml::from_str("[server]\nlisten=\"1.2.3.4:1\"").unwrap();
        assert_eq!(cfg.runner.python, "python3");
        assert_eq!(cfg.disk.keep_last_uploads, 3);
    }

    #[test]
    fn patch_updates_tokens() {
        let mut cfg = Config::default();
        let patch = serde_json::json!({"console_token": "abc", "instance_uuid": "p1-x"});
        cfg.apply_patch(&patch);
        assert_eq!(cfg.autodl.console_token, "abc");
        assert_eq!(cfg.autodl.instance_uuid, "p1-x");
        assert!(cfg.autodl.power_off_after_upload); // untouched
    }
}
