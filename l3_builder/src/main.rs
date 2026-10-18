//! Build the Instinct UltraData-Code-L3 12B-token CPT corpus.
//!
//! One parquet shard is one rayon task. Per record: text validity, global
//! exact dedup (blake2b-16), cross-corpus dedup (blake2b-8 against a sorted
//! fingerprint array), project-tokenizer length filter (<= 4096 tokens
//! including BOS/EOS), 13-word n-gram benchmark screening, vendor-mention
//! diversion, and a strict per-language token quota. Survivors are written as
//! JSONL; the final global shuffle, identity screening of diverted records,
//! and reporting stay in the Python builder.
//!
//! Memory is bounded on purpose: fingerprints live in sorted arrays (binary
//! search, not hash tables), tokenizers are thread-local so no shared cache
//! lock serializes workers, and each task holds only its current row-group
//! batch (~40 MB) with encodings dropped per record.
//!
//! Fingerprint conventions (must match the Python builder):
//!   fp16     = blake2b(text, 16 bytes)                 intra-corpus dedup
//!   fp8      = u64 (BE) of blake2b(text, 8 bytes)      cross-corpus dedup
//!   ngram fp = u64 (BE) of blake2b(join("\x1f", 13 words), 8 bytes)
//!   words(text) = matches of "[A-Za-z_]\w*|\d+" over text.lower()

use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{bail, Context as AnyhowContext, Result};
use arrow_array::{Array, LargeStringArray, StringArray};
use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use regex::Regex;
use serde::Deserialize;
use serde_json::json;
use tokenizers::Tokenizer;

const MIXTURE: &str = "ultradata-code-l3-language-proportional";
const LICENSE: &str = "apache-2.0";
const SOURCE_PREFIX: &str = "ultradata_code_l3";
const MAX_TOKENS: u64 = 4096;
const WORD_SEPARATOR: char = '\x1f';
const WORD_PATTERN: &str = r"[A-Za-z_]\w*|\d+";

fn word_regex() -> &'static Regex {
    static WORD_RE: OnceLock<Regex> = OnceLock::new();
    WORD_RE.get_or_init(|| Regex::new(WORD_PATTERN).expect("valid word pattern"))
}

static TOKENIZER_PATH: OnceLock<PathBuf> = OnceLock::new();

fn thread_tokenizer() -> &'static Tokenizer {
    use std::cell::OnceCell;
    thread_local! {
        static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();
    }
    TOKENIZER.with(|cell| {
        cell.get_or_init(|| {
            let path = TOKENIZER_PATH.get().expect("tokenizer path configured");
            Tokenizer::from_file(path).expect("failed to load tokenizer.json")
        })
    })
}

#[derive(Parser)]
struct Args {
    /// plan.json written by the Python builder
    #[arg(long)]
    plan: PathBuf,
    /// working dir that contains raw/<upstream path> shards
    #[arg(long)]
    work: PathBuf,
    /// sorted little-endian u64 cross-corpus fingerprints
    #[arg(long)]
    fingerprints: PathBuf,
    /// sorted little-endian u64 benchmark 13-gram fingerprints
    #[arg(long)]
    bench_ngrams: PathBuf,
    /// vendor_triggers.json written by the Python builder
    #[arg(long)]
    triggers: PathBuf,
    /// tokenizer.json of the project tokenizer
    #[arg(long)]
    tokenizer: PathBuf,
    /// accepted-records JSONL (pre-shuffle)
    #[arg(long)]
    out: PathBuf,
    /// diverted vendor-mention records for Python identity screening
    #[arg(long)]
    vendor_hits: PathBuf,
    /// per-language stats JSON
    #[arg(long)]
    stats: PathBuf,
    /// rayon threads (0 = all cores)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// process only this language (combined with --append to extend a run)
    #[arg(long)]
    only_lang: Option<String>,
    /// append to existing out/vendor files instead of refusing
    #[arg(long)]
    append: bool,
}

#[derive(Deserialize)]
struct Plan {
    languages: BTreeMap<String, LanguagePlan>,
}

#[derive(Deserialize)]
struct LanguagePlan {
    quota: u64,
    chosen_shards: Vec<ShardRef>,
}

#[derive(Deserialize)]
struct ShardRef {
    path: String,
}

#[derive(Deserialize)]
struct Triggers {
    ascii: Vec<String>,
    cjk: Vec<String>,
}

#[derive(Default)]
struct LangDelta {
    shards: u64,
    scanned: u64,
    invalid_text: u64,
    duplicate_text: u64,
    existing_corpus_duplicate: u64,
    outside_token_range: u64,
    benchmark_overlap: u64,
    vendor_hits: u64,
    over_budget: u64,
    accepted_rows: u64,
    accepted_tokens: u64,
    accepted_bytes: u64,
}

impl LangDelta {
    fn add(&mut self, other: &LangDelta) {
        self.shards += other.shards;
        self.scanned += other.scanned;
        self.invalid_text += other.invalid_text;
        self.duplicate_text += other.duplicate_text;
        self.existing_corpus_duplicate += other.existing_corpus_duplicate;
        self.outside_token_range += other.outside_token_range;
        self.benchmark_overlap += other.benchmark_overlap;
        self.vendor_hits += other.vendor_hits;
        self.over_budget += other.over_budget;
        self.accepted_rows += other.accepted_rows;
        self.accepted_tokens += other.accepted_tokens;
        self.accepted_bytes += other.accepted_bytes;
    }
}

/// Sharded mutex-protected set keeps intra-corpus dedup off one lock.
struct ShardedSet {
    shards: Vec<Mutex<HashSet<u64>>>,
    mask: u64,
}

impl ShardedSet {
    fn new(shard_bits: usize) -> Self {
        let shards = (0..(1usize << shard_bits))
            .map(|_| Mutex::new(HashSet::new()))
            .collect();
        Self { shards, mask: (1u64 << shard_bits) - 1 }
    }

    /// Returns false when the fingerprint was already present.
    fn insert(&self, fingerprint: u64) -> bool {
        let index = (fingerprint & self.mask) as usize;
        let mut guard = self.shards[index].lock().unwrap();
        !guard.insert(fingerprint)
    }
}

/// Buffered shared writer; per-task staging keeps the lock off the hot loop.
struct SharedWriter {
    inner: Mutex<BufWriter<File>>,
}

impl SharedWriter {
    fn open(path: &Path, append: bool) -> Result<Self> {
        if path.exists() && !append {
            bail!("refusing to overwrite existing file {}: pass --append", path.display());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { inner: Mutex::new(BufWriter::with_capacity(8 << 20, file)) })
    }

    fn write_chunk(&self, bytes: &[u8]) -> Result<()> {
        let mut guard = self.inner.lock().unwrap();
        guard.write_all(bytes)?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        self.inner.lock().unwrap().flush()?;
        Ok(())
    }
}

struct Task {
    language: usize,
    path: PathBuf,
    display: String,
}

struct Context {
    cross: Box<[u64]>,
    bench: Box<[u64]>,
    ascii_triggers: Vec<String>,
    cjk_regex: Option<Regex>,
    dedup: ShardedSet,
    quota_tokens: Vec<AtomicU64>,
    quotas: Vec<u64>,
    accepted_writer: Arc<SharedWriter>,
    vendor_writer: Arc<SharedWriter>,
}

fn blake2b(data: &[u8], size: usize) -> Vec<u8> {
    let mut hasher = Blake2bVar::new(size).expect("valid digest size");
    hasher.update(data);
    let mut out = vec![0u8; size];
    hasher.finalize_variable(&mut out).expect("finalize");
    out
}

fn blake2b_u64(data: &[u8]) -> u64 {
    let digest = blake2b(data, 8);
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest);
    u64::from_be_bytes(bytes)
}

fn read_u64_array(path: &Path) -> Result<Box<[u64]>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read {}", path.display()))?;
    if bytes.len() % 8 != 0 {
        bail!("{} is not a multiple of 8 bytes", path.display());
    }
    let values: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    Ok(values.into_boxed_slice())
}

fn load_sorted_u64(path: &Path) -> Result<Box<[u64]>> {
    let mut values = read_u64_array(path)?;
    if !values.is_sorted() {
        values.sort_unstable();
    }
    Ok(values)
}

fn contains_sorted(values: &[u64], needle: u64) -> bool {
    let index = values.partition_point(|value| *value < needle);
    index < values.len() && values[index] == needle
}

fn fingerprint_ngram(words: &[&str]) -> u64 {
    // Same shape as the Python builder: blake2b over "\x1f".join(words).
    let mut joined = Vec::with_capacity(16 * words.len());
    for (index, word) in words.iter().enumerate() {
        if index > 0 {
            joined.push(WORD_SEPARATOR as u8);
        }
        joined.extend_from_slice(word.as_bytes());
    }
    blake2b_u64(&joined)
}

fn benchmark_hit(text: &str, bench: &[u64]) -> bool {
    let lowered = text.to_lowercase();
    let words: Vec<&str> = word_regex().find_iter(&lowered).map(|m| m.as_str()).collect();
    if words.len() < 13 {
        return false;
    }
    words.windows(13).any(|window| contains_sorted(bench, fingerprint_ngram(window)))
}

fn vendor_hit(lowered: &str, text: &str, ctx: &Context) -> bool {
    if ctx.ascii_triggers.iter().any(|trigger| lowered.contains(trigger)) {
        return true;
    }
    if let Some(regex) = &ctx.cjk_regex {
        if regex.is_match(text) {
            return true;
        }
    }
    false
}

fn string_value<'a>(column: &'a dyn Array, row: usize) -> Result<&'a str> {
    if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(array.value(row));
    }
    if let Some(array) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(array.value(row));
    }
    bail!("unexpected string column type")
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn process_shard(task: &Task, ctx: &Context) -> Result<LangDelta> {
    let tokenizer = thread_tokenizer();
    let file = File::open(&task.path)
        .with_context(|| format!("open shard {}", task.path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read footer {}", task.path.display()))?;
    let reader = builder.build()?;
    let mut delta = LangDelta { shards: 1, ..LangDelta::default() };
    let mut out_buffer: Vec<u8> = Vec::with_capacity(1 << 20);
    let mut vendor_buffer: Vec<u8> = Vec::with_capacity(1 << 16);
    for batch in reader {
        let batch = batch.context("read record batch")?;
        let uuid_index = batch.schema().index_of("uuid")?;
        let text_index = batch.schema().index_of("full_content")?;
        let uuids = batch.column(uuid_index);
        let texts = batch.column(text_index);
        for row in 0..batch.num_rows() {
            delta.scanned += 1;
            if texts.is_null(row) {
                delta.invalid_text += 1;
                continue;
            }
            let text = string_value(texts, row)
                .with_context(|| format!("{} row {}", task.display, row))?;
            if text.trim().is_empty() {
                delta.invalid_text += 1;
                continue;
            }
            let uuid = if uuids.is_null(row) {
                String::new()
            } else {
                string_value(uuids, row)?.to_string()
            };
            let fp16 = blake2b(text.as_bytes(), 16);
            let mut intra = [0u8; 8];
            intra.copy_from_slice(&fp16[..8]);
            if !ctx.dedup.insert(u64::from_be_bytes(intra)) {
                delta.duplicate_text += 1;
                continue;
            }
            let fp8 = blake2b_u64(text.as_bytes());
            if contains_sorted(&ctx.cross, fp8) {
                delta.existing_corpus_duplicate += 1;
                continue;
            }
            let encoding = tokenizer
                .encode(text, false)
                .with_context(|| format!("tokenize {} row {}", task.display, row))?;
            let token_count = encoding.get_token_ids().len() as u64 + 2;
            if token_count < 1 || token_count > MAX_TOKENS {
                delta.outside_token_range += 1;
                continue;
            }
            if benchmark_hit(text, &ctx.bench) {
                delta.benchmark_overlap += 1;
                continue;
            }
            let lowered = text.to_lowercase();
            let diverted = vendor_hit(&lowered, text, ctx);
            let quota = ctx.quotas[task.language];
            let previous =
                ctx.quota_tokens[task.language].fetch_add(token_count, Ordering::Relaxed);
            if previous + token_count > quota {
                ctx.quota_tokens[task.language].fetch_sub(token_count, Ordering::Relaxed);
                delta.over_budget += 1;
                continue;
            }
            delta.accepted_rows += 1;
            delta.accepted_tokens += token_count;
            delta.accepted_bytes += text.len() as u64;
            if diverted {
                delta.vendor_hits += 1;
                let payload = json!({
                    "language": task.display,
                    "source_shard": task.display,
                    "uuid": uuid,
                    "text": text,
                    "token_count": token_count,
                    "fp16": hex(&fp16),
                    "fp8": fp8,
                });
                serde_json::to_writer(&mut vendor_buffer, &payload)?;
                vendor_buffer.push(b'\n');
                if vendor_buffer.len() >= (1 << 20) {
                    ctx.vendor_writer.write_chunk(&vendor_buffer)?;
                    vendor_buffer.clear();
                }
            } else {
                let row = json!({
                    "text": text,
                    "token_count": token_count,
                    "source": format!("{SOURCE_PREFIX}:{}", task.display),
                    "license": LICENSE,
                    "mix_category": task.display,
                    "mixture": MIXTURE,
                    "uuid": uuid,
                });
                serde_json::to_writer(&mut out_buffer, &row)?;
                out_buffer.push(b'\n');
                if out_buffer.len() >= (1 << 20) {
                    ctx.accepted_writer.write_chunk(&out_buffer)?;
                    out_buffer.clear();
                }
            }
        }
    }
    if !out_buffer.is_empty() {
        ctx.accepted_writer.write_chunk(&out_buffer)?;
    }
    if !vendor_buffer.is_empty() {
        ctx.vendor_writer.write_chunk(&vendor_buffer)?;
    }
    Ok(delta)
}

#[derive(serde::Serialize, Default, Clone, Copy)]
struct LangStat {
    quota: u64,
    shards: u64,
    scanned: u64,
    invalid_text: u64,
    duplicate_text: u64,
    existing_corpus_duplicate: u64,
    outside_token_range: u64,
    benchmark_overlap: u64,
    vendor_hits: u64,
    over_budget: u64,
    accepted_rows: u64,
    accepted_tokens: u64,
    accepted_bytes: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(if args.threads == 0 { 0 } else { args.threads })
        .build_global()?;
    TOKENIZER_PATH
        .set(args.tokenizer.clone())
        .map_err(|_| anyhow::anyhow!("tokenizer path already set"))?;

    let plan_text = std::fs::read_to_string(&args.plan)
        .with_context(|| format!("read {}", args.plan.display()))?;
    let plan: Plan = serde_json::from_str(&plan_text).context("parse plan.json")?;
    let triggers: Triggers = serde_json::from_str(
        &std::fs::read_to_string(&args.triggers).context("read vendor triggers")?,
    )
    .context("parse vendor triggers")?;

    let mut tasks: Vec<Task> = Vec::new();
    let mut quotas: Vec<u64> = Vec::new();
    let mut language_names: Vec<String> = Vec::new();
    for (language, value) in plan.languages.iter() {
        if let Some(only) = &args.only_lang {
            if language != only {
                continue;
            }
        }
        let index = language_names.len();
        language_names.push(language.clone());
        quotas.push(value.quota);
        for shard in &value.chosen_shards {
            let raw = args.work.join("raw").join(&shard.path);
            if !raw.is_file() {
                bail!("missing shard file {}", raw.display());
            }
            tasks.push(Task {
                language: index,
                path: raw,
                display: shard.path.clone(),
            });
        }
    }
    if tasks.is_empty() {
        bail!("no shards selected (check --only-lang)");
    }

    let cross = load_sorted_u64(&args.fingerprints)?;
    eprintln!("[l3] {} cross-corpus fingerprints loaded", cross.len());
    let bench = load_sorted_u64(&args.bench_ngrams)?;
    eprintln!("[l3] {} benchmark 13-gram fingerprints loaded", bench.len());
    let cjk_regex = if triggers.cjk.is_empty() {
        None
    } else {
        Some(Regex::new(
            &triggers.cjk.iter().map(|t| regex::escape(t)).collect::<Vec<_>>().join("|"),
        )?)
    };

    let context = Arc::new(Context {
        cross,
        bench,
        ascii_triggers: triggers.ascii.iter().map(|t| t.to_lowercase()).collect(),
        cjk_regex,
        dedup: ShardedSet::new(6),
        quota_tokens: quotas.iter().map(|_| AtomicU64::new(0)).collect(),
        quotas,
        accepted_writer: Arc::new(SharedWriter::open(&args.out, args.append)?),
        vendor_writer: Arc::new(SharedWriter::open(&args.vendor_hits, args.append)?),
    });

    let completed = AtomicUsize::new(0);
    let total = tasks.len();
    let results: Vec<(usize, Result<LangDelta>)> = tasks
        .into_par_iter()
        .map(|task| {
            let language = task.language;
            let outcome = process_shard(&task, &context);
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 5 == 0 || done == total {
                eprintln!("[l3] {done}/{total} shards done");
            }
            (language, outcome)
        })
        .collect();

    let mut failures = 0usize;
    let mut per_language: Vec<LangDelta> = vec![LangDelta::default(); language_names.len()];
    for (language, outcome) in results {
        match outcome {
            Ok(delta) => per_language[language].add(&delta),
            Err(error) => {
                failures += 1;
                eprintln!("[l3] shard failed: {error:#}");
            }
        }
    }
    if failures > 0 {
        bail!("{failures} shards failed; see stderr above");
    }

    context.accepted_writer.flush()?;
    context.vendor_writer.flush()?;

    let mut stats: BTreeMap<String, LangStat> = BTreeMap::new();
    for (index, name) in language_names.iter().enumerate() {
        let delta = &per_language[index];
        stats.insert(
            name.clone(),
            LangStat {
                quota: quotas[index],
                shards: delta.shards,
                scanned: delta.scanned,
                invalid_text: delta.invalid_text,
                duplicate_text: delta.duplicate_text,
                existing_corpus_duplicate: delta.existing_corpus_duplicate,
                outside_token_range: delta.outside_token_range,
                benchmark_overlap: delta.benchmark_overlap,
                vendor_hits: delta.vendor_hits,
                over_budget: delta.over_budget,
                accepted_rows: delta.accepted_rows,
                accepted_tokens: delta.accepted_tokens,
                accepted_bytes: delta.accepted_bytes,
            },
        );
    }
    let totals = stats.values().fold(LangStat::default(), |mut acc, stat| {
        acc.add_from(stat);
        acc
    });
    let threads_used = if args.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        args.threads
    };
    let report = json!({
        "languages": stats,
        "totals": totals,
        "tokenizer": args.tokenizer,
        "threads": threads_used,
        "text_field": "full_content",
        "max_tokens_including_bos_eos": MAX_TOKENS,
        "vendor_diversion": ("vendor-mention rows consume quota but are written to the "
                            .to_string() + "sidecar for Python identity screening"),
        "append_mode": args.append,
    });
    std::fs::write(&args.stats, serde_json::to_string_pretty(&report)? + "\n")
        .with_context(|| format!("write {}", args.stats.display()))?;
    println!(
        "[l3] accepted {} rows / {} tokens; stats -> {}",
        totals.accepted_rows,
        totals.accepted_tokens,
        args.stats.display(),
    );
    Ok(())
}

impl LangStat {
    fn add_from(&mut self, other: &LangStat) {
        self.quota += other.quota;
        self.shards += other.shards;
        self.scanned += other.scanned;
        self.invalid_text += other.invalid_text;
        self.duplicate_text += other.duplicate_text;
        self.existing_corpus_duplicate += other.existing_corpus_duplicate;
        self.outside_token_range += other.outside_token_range;
        self.benchmark_overlap += other.benchmark_overlap;
        self.vendor_hits += other.vendor_hits;
        self.over_budget += other.over_budget;
        self.accepted_rows += other.accepted_rows;
        self.accepted_tokens += other.accepted_tokens;
        self.accepted_bytes += other.accepted_bytes;
    }
}
