//! Translate config_webui's current panel state, without applying presets or stale pipeline snapshots.
use crate::config::Config;
use anyhow::{bail, Context};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

const STAGES: &[&str] = &[
    "pretrain",
    "cpt",
    "full_sft",
    "lora",
    "dpo",
    "ppo",
    "grpo",
    "agent",
    "distillation",
];

fn get(state: &Value, key: &str, default: Value) -> Value {
    state
        .get(key)
        .filter(|value| !value.is_null())
        .cloned()
        .unwrap_or(default)
}

fn enabled(state: &Value, key: &str, default: bool) -> bool {
    state
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .unwrap_or_else(|| value.as_i64().unwrap_or(0) != 0)
        })
        .unwrap_or(default)
}

fn vector(state: &Value, key: &str, count: usize, default: f64) -> anyhow::Result<Value> {
    let raw = state.get(key);
    let mut value = match raw {
        Some(Value::Array(_)) => raw.unwrap().clone(),
        Some(Value::String(text)) if !text.trim().is_empty() => {
            if text.trim().starts_with('[') {
                serde_json::from_str(text)?
            } else {
                Value::Array(
                    text.split(',')
                        .filter(|part| !part.trim().is_empty())
                        .map(|part| part.trim().parse::<f64>().map(|number| json!(number)))
                        .collect::<Result<_, _>>()?,
                )
            }
        }
        _ => json!(vec![default; count]),
    };
    if value.as_array().is_some_and(|values| values.is_empty()) {
        value = json!(vec![default; count]);
    }
    if value.as_array().is_none_or(|values| {
        values.len() != count || values.iter().any(|item| item.as_f64().is_none())
    }) {
        bail!("缓存 {key} 的 LongRoPE 向量长度或数值无效，应为 {count} 个数");
    }
    Ok(value)
}

fn model(state: &Value) -> anyhow::Result<Value> {
    let hidden = state["hidden_size"]
        .as_u64()
        .filter(|number| *number > 0)
        .context("缓存缺少有效 hidden_size")?;
    let layers = state["num_hidden_layers"]
        .as_u64()
        .filter(|number| *number > 0)
        .context("缓存缺少有效 num_hidden_layers")?;
    let heads = get(state, "num_attention_heads", json!(8))
        .as_u64()
        .filter(|number| *number > 0)
        .context("缓存注意力头数无效")?;
    let intermediate = ((hidden as f64 * std::f64::consts::PI / 64.0).ceil() as u64) * 64;
    let mut result =
        json!({"hidden_size":hidden,"num_hidden_layers":layers,"head_dim":hidden/heads});
    for (key, default) in [
        ("vocab_size", json!(6400)),
        ("dropout", json!(0.0)),
        ("hidden_act", json!("silu")),
        ("tie_word_embeddings", json!(false)),
        ("num_attention_heads", json!(8)),
        ("num_key_value_heads", json!(4)),
        ("max_position_embeddings", json!(32768)),
        ("rope_theta", json!(1e6)),
        ("use_moe", json!(false)),
        ("use_grad_checkpoint", json!(0)),
        ("rms_norm_eps", json!(1e-6)),
        ("flash_attn", json!(true)),
        ("intermediate_size", json!(intermediate)),
        ("residual_type", json!("standard")),
        ("model_architecture", json!("standard")),
    ] {
        result[key] = get(state, key, default);
    }
    let scaling = state
        .get("rope_scaling_type")
        .and_then(Value::as_str)
        .unwrap_or(if enabled(state, "inference_rope_scaling", false) {
            "yarn"
        } else {
            "none"
        });
    result["inference_rope_scaling"] = json!(scaling != "none");
    if scaling == "yarn" {
        let mut rope = json!({"type":"yarn"});
        for (key, default) in [
            ("beta_fast", json!(32)),
            ("beta_slow", json!(1)),
            ("factor", json!(16)),
            ("original_max_position_embeddings", json!(2048)),
            ("attention_factor", json!(1.0)),
        ] {
            rope[key] = get(state, key, default);
        }
        result["rope_scaling"] = rope;
    } else if scaling == "longrope" {
        let original = get(state, "original_max_position_embeddings", json!(4096))
            .as_f64()
            .filter(|number| *number > 0.0)
            .context("缓存原始上下文长度无效")?;
        let factor = get(
            state,
            "factor",
            json!(
                result["max_position_embeddings"]
                    .as_f64()
                    .unwrap_or(32768.0)
                    / original
            ),
        )
        .as_f64()
        .context("缓存 RoPE factor 无效")?;
        let mut rope = json!({"type":"longrope","factor":factor,"original_max_position_embeddings":original as u64,
            "short_factor":vector(state,"longrope_short_factors",(hidden/heads/2) as usize,1.0)?,
            "long_factor":vector(state,"longrope_long_factors",(hidden/heads/2) as usize,factor)?});
        for (source, target, default) in [
            (
                "longrope_short_attention_factor",
                "short_attention_factor",
                json!(1.0),
            ),
            (
                "longrope_short_retained_tokens",
                "short_retained_start_tokens",
                json!(0),
            ),
            (
                "longrope_long_retained_tokens",
                "long_retained_start_tokens",
                json!(0),
            ),
        ] {
            rope[target] = get(state, source, default);
        }
        let attention = get(state, "longrope_long_attention_factor", json!(0.0));
        if attention.as_f64().unwrap_or(0.0) > 0.0 {
            rope["long_attention_factor"] = attention;
        }
        result["rope_scaling"] = rope;
    }
    if enabled(state, "early_exit_enabled", false) {
        result["early_exit_layers"] = get(state, "early_exit_layers", json!([4, 5, 6, 7]));
        result["early_exit_loss_weight"] = get(state, "early_exit_loss_weight", json!(0.3));
    }
    match result["model_architecture"].as_str().unwrap_or("standard") {
        "linear" => {
            for (key, default) in [
                ("full_attention_interval", 4),
                ("linear_conv_kernel_dim", 4),
                ("linear_key_head_dim", hidden / heads),
                ("linear_value_head_dim", hidden / heads),
                ("linear_num_key_heads", heads),
                ("linear_num_value_heads", heads),
            ] {
                result[key] = get(state, key, json!(default));
            }
        }
        "looped" => {
            result["recurrent_architecture_version"] = json!(2);
            for (key, default) in [
                ("qk_bias", json!(true)),
                ("qk_norm", json!(false)),
                ("prelude_layers", json!(2)),
                ("recurrent_layers", json!(4)),
                ("coda_layers", json!(2)),
                ("loop_iters", json!(8)),
                ("mean_backprop_depth", json!(4)),
                ("recurrence_sampling", json!("lognormal_poisson")),
                ("recurrence_log_normal_sigma", json!(1.0)),
                ("max_recurrence", json!(32)),
                ("state_init_std", json!((2.0f64 / 5.0).sqrt())),
                ("embedding_scale", json!((hidden as f64).sqrt())),
                ("use_input_injection", json!(true)),
            ] {
                result[key] = get(state, key, default);
            }
        }
        _ => {}
    }
    match result["residual_type"].as_str().unwrap_or("standard") {
        "mhc" => {
            for (key, default) in [
                ("hc_mult", json!(4)),
                ("hc_sinkhorn_iters", json!(20)),
                ("hc_eps", json!(1e-6)),
            ] {
                result[key] = get(state, key, default);
            }
        }
        "attnres" => {
            result["attnres_variant"] = get(state, "attnres_variant", json!("block"));
            result["attnres_block_size"] = get(
                state,
                "attnres_block_size",
                json!((2 * layers).div_ceil(8).max(1)),
            );
        }
        _ => {}
    }
    if enabled(state, "use_moe", false) {
        for (key, default) in [
            ("num_experts", json!(4)),
            ("num_experts_per_tok", json!(1)),
            ("moe_intermediate_size", json!(intermediate)),
            ("norm_topk_prob", json!(true)),
            ("router_aux_loss_coef", json!(5e-4)),
        ] {
            result[key] = get(state, key, default);
        }
    }
    Ok(result)
}

fn portable_path(value: &str) -> String {
    let normalized = value.replace('\\', "/");
    if let Some(tail) = normalized.strip_prefix("//?/UNC/") {
        format!("//{tail}")
    } else {
        normalized
            .strip_prefix("//?/")
            .unwrap_or(&normalized)
            .to_string()
    }
}

pub(crate) fn data_path(value: &str, local_root: &str, remote_root: &str) -> String {
    let normalized = portable_path(value);
    for root in [local_root, remote_root] {
        let root = portable_path(root).trim_end_matches('/').to_string() + "/";
        let matches = if root.as_bytes().get(1) == Some(&b':') || root.starts_with("//") {
            normalized.to_lowercase().starts_with(&root.to_lowercase())
        } else {
            normalized.starts_with(&root)
        };
        if matches {
            return normalized[root.len()..].to_string();
        }
    }
    normalized.trim_start_matches("./").to_string()
}

pub(crate) fn profile(
    state: &Value,
    stage: &str,
    local_root: &str,
    remote_root: &str,
) -> anyhow::Result<Value> {
    if !STAGES.contains(&stage) {
        bail!("缓存训练阶段不受支持：{stage}");
    }
    let model = model(state)?;
    let mut args = Map::new();
    // Same field names and defaults as config_webui._build_training_command.
    for (key, default) in [
        ("batch_size", json!(32)),
        ("max_seq_len", json!(768)),
        ("accumulation_steps", json!(1)),
        ("optimizer", json!("adamw")),
        ("sequence_packing_mode", json!("fixed")),
        ("seq_bucket", json!(2)),
        ("bucket_gpu_memory_gb", json!(16.0)),
        ("bucket_max_seq_len", json!(16384)),
        ("bucket_large_threshold", json!(8192)),
        ("packing_batch_size", json!(1000)),
        ("packing_num_proc", json!(if cfg!(windows) { 4 } else { 8 })),
        ("bucket_loader_workers", json!(0)),
        ("data_cache_max_gb", json!(5.0)),
        ("dataset_streaming", json!("auto")),
        ("streaming_chunk_mb", json!(1024)),
        ("cache_build_mode", json!("inline")),
        ("param_dtype", json!("fp32")),
        ("kv_cache_dtype", json!("fp32")),
        ("fp8_training", json!("off")),
        ("fp8_filter", json!("auto")),
        ("profile", json!("off")),
        ("profile_warmup", json!(10)),
        ("profile_interval", json!(100)),
        ("profile_active_steps", json!(5)),
        ("compile_mode", json!("reduce-overhead")),
        ("use_grad_checkpoint", json!(0)),
    ] {
        args.insert(key.into(), get(state, key, default));
    }
    args.insert(
        "dtype".into(),
        get(state, "activation_dtype", json!("bfloat16")),
    );
    let defaults = match stage {
        "pretrain" => (2, 5e-4),
        "cpt" => (1, 3e-5),
        "full_sft" => (2, 1e-5),
        "lora" => (10, 1e-4),
        "dpo" => (1, 4e-8),
        "ppo" | "grpo" | "agent" => (1, 3e-7),
        _ => (6, 5e-6),
    };
    args.insert(
        "epochs".into(),
        get(state, &format!("epochs_{stage}"), json!(defaults.0)),
    );
    args.insert(
        "learning_rate".into(),
        get(state, &format!("learning_rate_{stage}"), json!(defaults.1)),
    );
    args.insert(
        "warmup_ratio".into(),
        get(
            state,
            &format!("warmup_ratio_{stage}"),
            json!(if stage == "cpt" { 0.03 } else { 0.0 }),
        ),
    );
    args.insert("min_lr_ratio".into(), json!(0.1));
    if state.get("bucket_token_budget").and_then(Value::as_u64).unwrap_or(0) > 0 {
        args.insert("bucket_token_budget".into(), state["bucket_token_budget"].clone());
    }
    if stage == "pretrain" && state["dataset_streaming"] == "on"
        && get(state, "cache_build_mode", json!("inline")) == "inline" {
        args.insert("num_workers".into(), json!(0));
    }
    if stage == "pretrain" && enabled(state, "autotune_use_profile", false) {
        let path = state["autotune_profile_path"].as_str().filter(|s| !s.trim().is_empty())
            .context("请选择服务器上的自动调优 profile 路径")?;
        args.insert("autotune_profile".into(), json!(data_path(path, local_root, remote_root)));
    }
    args.insert(
        "use_compile".into(),
        json!(enabled(state, "use_compile", true)),
    );
    args.insert(
        "streaming_prefetch_chunks".into(),
        json!(u8::from(enabled(state, "streaming_prefetch_chunks", true))),
    );
    args.insert(
        "sequence_packing".into(),
        json!(
            ["pretrain", "cpt", "full_sft", "lora", "distillation"].contains(&stage)
                && enabled(state, "sequence_packing", false)
        ),
    );
    if model["model_architecture"] == "looped" {
        args.insert("use_looped".into(), json!(true));
    }
    if model.get("early_exit_layers").is_some() {
        args.insert("early_exit".into(), json!(true));
    }
    if enabled(state, "moe_router_migration", false)
        && enabled(state, "use_moe", false)
        && ["pretrain", "cpt", "full_sft"].contains(&stage)
        && !enabled(state, "from_resume", false)
    {
        let top = model["num_experts_per_tok"]
            .as_u64()
            .context("缓存 top-k 无效")?;
        if top < 1 || top > model["num_experts"].as_u64().unwrap_or(0) {
            bail!("缓存路由迁移 top-k 无效");
        }
        args.insert("moe_router_top_k".into(), json!(top));
        args.insert(
            "moe_router_norm_topk_prob".into(),
            json!(u8::from(top > 1 && enabled(state, "norm_topk_prob", true))),
        );
    }
    if let Some(path) = state
        .get(format!("data_path_{stage}"))
        .and_then(Value::as_str)
    {
        args.insert(
            "data_path".into(),
            json!(data_path(path, local_root, remote_root)),
        );
    }
    Ok(
        json!({"trainer":stage,"args":args,"model_config":model,"resume_requested":enabled(state,"from_resume",false)}),
    )
}

pub fn load(cfg: &Config, config_path: &Path, stage: Option<&str>) -> anyhow::Result<Value> {
    let configured = Path::new(&cfg.training.panel_state_path);
    let path = if configured.is_absolute() {
        configured.to_path_buf()
    } else {
        config_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(configured)
    };
    if std::fs::metadata(&path)
        .map(|metadata| metadata.len() > 4 * 1024 * 1024)
        .unwrap_or(false)
    {
        bail!("config_webui 缓存超过 4 MiB");
    }
    let bytes = std::fs::read(&path).with_context(|| {
        format!(
            "无法读取 config_webui 缓存 {}，请配置 training.panel_state_path",
            path.display()
        )
    })?;
    if bytes.len() > 4 * 1024 * 1024 {
        bail!("config_webui 缓存超过 4 MiB");
    }
    let state: Value = serde_json::from_slice(&bytes)
        .context("config_webui 缓存 JSON 无效，请待面板保存完成后重试")?;
    if !state.is_object() {
        bail!("config_webui 缓存必须是 JSON 对象");
    }
    let canonical = path.canonicalize()?;
    let root = canonical
        .parent()
        .and_then(Path::parent)
        .context("缓存应位于 trainer/ 下")?;
    let stage = stage.unwrap_or_else(|| state["train_type"].as_str().unwrap_or("pretrain"));
    let mut result = profile(
        &state,
        stage,
        &root.to_string_lossy(),
        &cfg.training.repo_root,
    )?;
    let parameter_bytes = serde_json::to_vec(&result)?;
    result["source"] = json!(path.display().to_string());
    result["sha256"] = json!(format!("{:x}", Sha256::digest(&parameter_bytes)));
    result["state_sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
    result["preset"] = state.get("preset").cloned().unwrap_or(Value::Null);
    result["panel_state"] = state;
    Ok(result)
}

pub fn apply(cfg: &Config, config_path: &Path, mut input: Value) -> anyhow::Result<Value> {
    if input.get("use_panel_cache") != Some(&json!(true)) {
        return Ok(input);
    }
    let cached = load(
        cfg,
        config_path,
        input.get("trainer").and_then(Value::as_str),
    )?;
    if let Some(expected) = input.get("panel_cache_sha256").and_then(Value::as_str) {
        if expected != cached["sha256"].as_str().unwrap() {
            bail!("config_webui 缓存已变化，请重新读取缓存，核对后再启动");
        }
    }
    input["trainer"] = cached["trainer"].clone();
    let mut args = cached["args"].as_object().unwrap().clone();
    if let Some(overrides) = input.get("args") {
        args.extend(
            overrides
                .as_object()
                .context("训练参数必须是 JSON 对象")?
                .clone(),
        );
    }
    input["args"] = Value::Object(args);
    if input["config_path"].is_null() || input["config_path"] == "__panel_cache__" {
        input["model_config"] = cached["model_config"].clone();
        input["config_path"] = json!("__panel_cache__");
    }
    input["panel_cache_source"] = json!({"path":cached["source"],"sha256":cached["sha256"],"state_sha256":cached["state_sha256"]});
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn latest_panel_values_win_over_pipeline_and_windows_paths_are_translated() {
        let state = json!({"hidden_size":512,"num_hidden_layers":32,"use_moe":true,"batch_size":1,"accumulation_steps":1,
            "optimizer":"muon","activation_dtype":"bfloat16","param_dtype":"fp32","fp8_training":"tensorwise",
            "use_compile":true,"compile_mode":"max-autotune-no-cudagraphs","learning_rate_full_sft":1e-9,
            "data_path_pretrain":"D:\\AI\\Instinct\\dataset\\pretrain_clean.parquet",
            "pipeline_stages":{"pretrain":{"args":{"batch_size":8,"param_dtype":"fp16"}}}});
        let pretrain = profile(&state, "pretrain", "D:/AI/Instinct", "/root/autodl-tmp").unwrap();
        assert_eq!(pretrain["args"]["batch_size"], 1);
        assert_eq!(pretrain["args"]["param_dtype"], "fp32");
        assert_eq!(
            pretrain["args"]["data_path"],
            "dataset/pretrain_clean.parquet"
        );
        assert_eq!(
            pretrain["args"]["compile_mode"],
            "max-autotune-no-cudagraphs"
        );
        assert_eq!(
            profile(&state, "full_sft", "D:/AI/Instinct", "/root/autodl-tmp").unwrap()["args"]
                ["learning_rate"],
            1e-9
        );
        assert!(pretrain["args"].get("from_resume").is_none());
    }
    #[test]
    fn rope_and_architecture_follow_the_current_panel() {
        let state = json!({"hidden_size":512,"num_hidden_layers":32,"num_attention_heads":16,"rope_scaling_type":"longrope",
            "longrope_short_factors":vec![1.0;16],"longrope_long_factors":vec![8.0;16],"model_architecture":"looped","loop_iters":12});
        let cached = model(&state).unwrap();
        assert_eq!(cached["rope_scaling"]["long_factor"][0], 8.0);
        assert_eq!(cached["head_dim"], 32);
        assert_eq!(cached["loop_iters"], 12);
        let mut bad = state.clone();
        bad["longrope_long_factors"] = json!("[1,2]");
        assert!(model(&bad).is_err());
    }
    #[test]
    fn canonical_windows_paths_keep_repo_relative_dataset_resolution() {
        assert_eq!(
            data_path(
                r"D:\AI\Instinct\dataset\pretrain_clean.parquet",
                r"\\?\D:\AI\Instinct",
                "/root/autodl-tmp"
            ),
            "dataset/pretrain_clean.parquet"
        );
        assert_eq!(
            data_path(
                r"\\server\repo\dataset\sft_clean.jsonl",
                r"\\?\UNC\server\repo",
                "/root/autodl-tmp"
            ),
            "dataset/sft_clean.jsonl"
        );
        assert_eq!(
            data_path(
                "/root/autodl-tmp/dataset/clean.parquet",
                "D:/AI/Instinct",
                "/root/autodl-tmp"
            ),
            "dataset/clean.parquet"
        );
        assert_eq!(
            data_path(
                "D:/elsewhere/data.parquet",
                "D:/AI/Instinct",
                "/root/autodl-tmp"
            ),
            "D:/elsewhere/data.parquet"
        );
    }
    #[test]
    fn cache_errors_and_changes_block_launch_and_explicit_overrides_survive() {
        let root = std::env::temp_dir().join(format!("relay-panel-cache-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("trainer")).unwrap();
        std::fs::create_dir_all(root.join("data_relay")).unwrap();
        let cfg = Config::default();
        let config_path = root.join("data_relay/config.toml");
        assert!(apply(&cfg, &config_path, json!({"use_panel_cache":true})).is_err());
        let state = root.join("trainer/webui_state.json");
        std::fs::write(
            &state,
            r#"{"train_type":"pretrain","hidden_size":512,"num_hidden_layers":32,"batch_size":1}"#,
        )
        .unwrap();
        let hash = load(&cfg, &config_path, None).unwrap()["sha256"].clone();
        let applied = apply(
            &cfg,
            &config_path,
            json!({"use_panel_cache":true,"args":{"batch_size":3},"panel_cache_sha256":hash}),
        )
        .unwrap();
        assert_eq!(applied["args"]["batch_size"], 3);
        assert_eq!(applied["model_config"]["hidden_size"], 512);
        std::fs::write(&state, r#"{"hidden_size":768,"num_hidden_layers":20}"#).unwrap();
        assert!(apply(
            &cfg,
            &config_path,
            json!({"use_panel_cache":true,"panel_cache_sha256":hash})
        )
        .is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
