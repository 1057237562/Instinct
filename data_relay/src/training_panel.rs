//! Editable relay panel. Cache import and interactive state share the same compiler.
use crate::{config::Config, runner::Relay, training_cache};
use anyhow::{bail, Context};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

pub fn schema() -> Value {
    serde_json::from_str(include_str!("training_schema.json")).expect("embedded training schema")
}

fn validate_value(value: &Value, field: &Value, key: &str) -> anyhow::Result<()> {
    if value.is_null() { return Ok(()); }
    let valid = match field["type"].as_str().unwrap_or("text") {
        "int" => value.as_f64().is_some_and(|n| n.fract() == 0.0),
        "float" => value.as_f64().is_some(),
        "bool" => value.is_boolean() || value == &json!(0) || value == &json!(1),
        "json" => value.is_array(),
        _ => value.is_string(),
    };
    if !valid { bail!("{key} 的类型无效"); }
    if let Some(choices) = field["choices"].as_array() {
        if !choices.contains(value) { bail!("{key} 不是支持的选项"); }
    }
    if let Some(number) = value.as_f64() {
        if field["min"].as_f64().is_some_and(|min| number < min) || field["max"].as_f64().is_some_and(|max| number > max) {
            bail!("{key} 超出有效范围");
        }
    }
    Ok(())
}

pub fn validate_state(state: &Value) -> anyhow::Result<()> {
    if !state.is_object() || state.to_string().len() > 262144 { bail!("面板配置必须是小于 256 KiB 的 JSON 对象"); }
    let schema = schema();
    for group in schema["groups"].as_array().unwrap() {
        for field in group["fields"].as_array().unwrap() {
            for stage in schema["stages"].as_array().unwrap() {
                let key = field["key"].as_str().unwrap().replace("{stage}", stage.as_str().unwrap());
                if let Some(value) = state.get(&key) { validate_value(value, field, &key)?; }
            }
        }
    }
    Ok(())
}

pub fn load(relay: &Relay) -> anyhow::Result<Value> {
    let path = relay.data_dir.join("training/panel.json");
    let (document, source) = if path.exists() {
        (serde_json::from_slice(&std::fs::read(path)?)?, "已保存的中继配置".to_string())
    } else if let Ok(cached) = training_cache::load(&relay.config(), &relay.config_path, None) {
        let mut state = cached["panel_state"].clone();
        if let Some(stage) = cached["trainer"].as_str() {
            state[format!("data_path_{stage}")] = cached["args"]["data_path"].clone();
        }
        (json!({"state":state,"extras":{},"launch":{}}), format!("导入 {}", cached["source"].as_str().unwrap_or("Config WebUI")))
    } else {
        let mut state = schema()["presets"]["Instinct V1 MoE"].clone();
        state["train_type"] = json!("pretrain");
        (json!({"state":state,"extras":{},"launch":{}}), "Instinct V1 MoE 默认配置".to_string())
    };
    Ok(json!({"document":document,"source":source,"schema":schema()}))
}

pub fn save(relay: &Relay, document: Value) -> anyhow::Result<Value> {
    validate_state(&document["state"])?;
    if document.to_string().len() > 524288 || !document["extras"].is_object() || !document["launch"].is_object() {
        bail!("面板存档格式无效或超过 512 KiB");
    }
    let path = relay.data_dir.join("training/panel.json");
    std::fs::create_dir_all(path.parent().unwrap())?;
    let tmp = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4().simple()));
    std::fs::write(&tmp, serde_json::to_vec_pretty(&document)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(json!({"saved":true}))
}

pub fn apply(cfg: &Config, config_path: &Path, mut input: Value) -> anyhow::Result<Value> {
    let Some(state) = input.get("panel_state") else { return Ok(input); };
    validate_state(state)?;
    let stage = input["trainer"].as_str().context("请选择训练阶段")?;
    let root = config_path.parent().unwrap_or(Path::new(".")).join(&cfg.training.local_repo_root);
    let root = root.canonicalize().unwrap_or(root);
    let compiled = training_cache::profile(state, stage, &root.to_string_lossy(), &cfg.training.repo_root)?;
    let mut args = compiled["args"].as_object().unwrap().clone();
    let schema = schema();
    if let Some(extra) = input.get("args") {
        for (key, value) in extra.as_object().context("额外训练参数必须是 JSON 对象")? {
            if ["data_path", "from_weight"].contains(&key.as_str()) {
                if !value.is_string() { bail!("{key} 必须是路径或权重名称"); }
            } else {
                let field = &schema["arguments"][stage][key];
                if field.is_null() { bail!("{key} 不属于该阶段的额外参数，请使用对应配置控件"); }
                validate_value(value, field, key)?;
            }
            if value.is_null() { bail!("{key} 不能为 null，请删除以使用默认值"); }
            let value = if key == "data_path" {
                json!(training_cache::data_path(value.as_str().unwrap(), &root.to_string_lossy(), &cfg.training.repo_root))
            } else { value.clone() };
            args.insert(key.clone(), value);
        }
    }
    let model = &compiled["model_config"];
    let hidden = model["hidden_size"].as_u64().unwrap();
    let heads = model["num_attention_heads"].as_u64().unwrap();
    let kv = model["num_key_value_heads"].as_u64().unwrap_or(0);
    if hidden % heads != 0 || kv == 0 || heads % kv != 0 { bail!("隐藏维度需整除 Q 头数，Q 头数需整除 KV 头数"); }
    if model["use_moe"] == true && model["num_experts_per_tok"].as_u64() > model["num_experts"].as_u64() {
        bail!("MoE Top-k 不能超过专家数");
    }
    input["args"] = Value::Object(args);
    if input["config_path"].is_null() || input["config_path"] == "__panel_cache__" {
        input["model_config"] = model.clone();
        input["config_path"] = json!("__panel_cache__");
    }
    // The submitted state is a snapshot; later cache edits cannot change a queued run.
    input["panel_cache_source"] = json!({"path":"relay-panel","sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&compiled)?))});
    input.as_object_mut().unwrap().remove("panel_state");
    input["use_panel_cache"] = json!(false);
    Ok(input)
}

fn quote(text: &str) -> String { format!("'{}'", text.replace('\'', "'\\''")) }

pub fn preview(relay: &Relay, input: Value) -> anyhow::Result<Value> {
    let cfg = relay.config();
    let result = apply(&cfg, &relay.config_path, input)?;
    let stage = result["trainer"].as_str().context("请选择训练阶段")?;
    if !schema()["stages"].as_array().unwrap().contains(&json!(stage)) { bail!("训练阶段无效"); }
    let mut args = result["args"].as_object().context("缺少训练参数")?.clone();
    let root = cfg.training.repo_root.trim_end_matches('/');
    let model = &result["model_config"];
    for key in ["hidden_size","num_hidden_layers","use_moe","use_grad_checkpoint","param_dtype","kv_cache_dtype"] {
        if let Some(value) = model.get(key) { args.entry(key).or_insert_with(|| value.clone()); }
    }
    if stage == "pretrain" { args.insert("from_weight".into(), json!("none")); }
    if stage == "distillation" {
        for role in ["student","teacher"] {
            for (suffix,source) in [("hidden_size","hidden_size"),("num_layers","num_hidden_layers"),("use_moe","use_moe")] {
                if let Some(value) = model.get(source) { args.entry(format!("{role}_{suffix}")).or_insert_with(||value.clone()); }
            }
        }
        args.entry("from_student_weight").or_insert(json!("none"));
        if let Some(base) = args.remove("from_weight") { args.entry("from_teacher_weight").or_insert(base); }
    }
    if let Some(path) = args.get("data_path").and_then(Value::as_str).filter(|s| !s.starts_with('/')).map(str::to_string) {
        args.insert("data_path".into(), json!(format!("{root}/{}",path.trim_start_matches("./"))));
    }
    let imported = result["existing_resume"].as_str().is_some_and(|s| !s.is_empty());
    let config = format!("{root}/.relay_training/<job-id>/model_config.json");
    args.insert("config_path".into(), json!(config));
    args.insert("save_dir".into(), json!(format!("{root}/out")));
    args.insert("pause_file".into(), json!(format!("{root}/.relay_training/<job-id>/pause.request")));
    args.insert("from_resume".into(), json!(u8::from(imported)));
    args.insert(if stage == "lora" {"lora_name"} else {"save_weight"}.into(), json!("<run-prefix>"));
    let count = result.get("nproc").and_then(Value::as_u64).unwrap_or(1);
    if !(1..=8).contains(&count) { bail!("GPU 数量必须为 1–8"); }
    let mut argv = vec![cfg.training.python.clone()];
    if count > 1 { argv.extend(["-m".into(),"torch.distributed.run".into(),"--nproc_per_node".into(),count.to_string(),"--master_port".into(),"<port>".into()]); }
    match model["model_architecture"].as_str() {
        Some("linear") => argv.push("run_linear.py".into()), Some("recurrsive_instinct") => argv.push("run_recurrsive.py".into()), _ => ()
    }
    argv.push(format!("trainer/train_{}.py",if stage == "cpt" {"pretrain"} else {stage}));
    for (key,value) in args {
        if ["use_wandb","debug_mode"].contains(&key.as_str()) {
            if value == true || value == 1 { argv.push(format!("--{key}")); }
        } else {
            argv.push(format!("--{key}"));
            argv.push(if let Some(b) = value.as_bool() {u8::from(b).to_string()} else if let Some(s) = value.as_str() {s.to_string()} else {value.to_string()});
        }
    }
    let devices = result["gpu_devices"].as_str().unwrap_or("");
    let prefix = if devices.is_empty() {String::new()} else {format!("CUDA_VISIBLE_DEVICES={} ",quote(devices))};
    let command = format!("cd {} && {prefix}{}",quote(root),argv.iter().map(|s|quote(s)).collect::<Vec<_>>().join(" "));
    Ok(json!({"request":result,"command":command,"argv":argv,"note":"任务 ID、输出前缀和 DDP 端口在启动时生成；检查点或服务器配置文件的结构参数及主干包装器在远端读取后确定。实际执行命令会写入任务日志。"}))
}
