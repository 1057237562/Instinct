//! Command-line surface.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::compile::{DEFAULT_BATCH_ROWS, DEFAULT_SCHEMA_ROWS};

#[derive(Debug, Parser)]
#[command(
    name = "dataset_compiler",
    version,
    about = "Compile Instinct JSONL corpora into Parquet",
    long_about = "Compile one JSONL corpus (or a directory of them) into Parquet that \
                  `datasets.load_dataset('parquet', ...)` and the Instinct trainers read \
                  directly.\n\n\
                  The output schema is inferred from the whole file by default, so a key \
                  discovered in the last row is still modelled in the first record batch. \
                  Chat data becomes list<struct<...>> with all-utf8 fields; everything else \
                  is typed, and values that cannot be typed are stored as JSON text so no \
                  row is ever dropped."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Compile JSONL into Parquet.
    Compile(CompileArgs),
    /// Print a parquet footer summary without reading data pages.
    Inspect(InspectArgs),
    /// Compare a compiled parquet file against its JSONL source.
    Verify(VerifyArgs),
}

#[derive(Debug, Args)]
pub struct CompileArgs {
    /// Source JSONL file, or a directory when the output is a directory.
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,
    /// Destination `.parquet` file, or a directory for a directory input.
    #[arg(short, long, value_name = "PATH")]
    pub output: PathBuf,
    /// Corpus family: auto, pretrain, sft, dpo, agent or generic.
    #[arg(long, default_value = "auto", value_name = "FORMAT")]
    pub format: String,
    /// Parquet compression: zstd, snappy, gzip, brotli or none.
    #[arg(long, default_value = "zstd", value_name = "CODEC")]
    pub compression: String,
    /// Codec level (zstd 1-22, gzip 0-10, brotli 0-11).
    #[arg(long, value_name = "N")]
    pub compression_level: Option<i32>,
    /// Rows per parquet row group; streaming chunks align to these.
    #[arg(long, default_value_t = 65_536, value_name = "N")]
    pub row_group_rows: usize,
    /// Byte ceiling per parquet row group, so long-row corpora stay seekable.
    #[arg(long, default_value_t = 128, value_name = "MB")]
    pub row_group_mb: usize,
    /// Cut row groups at the same rows the streaming planner cuts JSONL chunks
    /// of this many bytes (e.g. 1GiB), so a compiled corpus keeps a training
    /// cursor's meaning.
    #[arg(long, value_parser = parse_size, value_name = "SIZE")]
    pub align_chunk_bytes: Option<u64>,
    /// Rows per record batch.
    #[arg(long, default_value_t = DEFAULT_BATCH_ROWS, value_name = "N")]
    pub batch_rows: usize,
    /// Parse threads (0 = one per available core).
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub threads: usize,
    /// Scan the whole file for the exact schema (false samples --schema-rows).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub pre_scan: bool,
    /// Rows sampled for the schema when --pre-scan false.
    #[arg(long, default_value_t = DEFAULT_SCHEMA_ROWS, value_name = "N")]
    pub schema_rows: usize,
    /// Dictionary encoding: auto (off for a text column), on or off.
    #[arg(long, default_value = "auto", value_name = "MODE")]
    pub dictionary: String,
    /// Write min/max statistics (off: corpora hold multi-kilobyte strings).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub statistics: bool,
    /// Column holding pretrain text.
    #[arg(long, default_value = "text", value_name = "NAME")]
    pub text_key: String,
    /// Stop after this many source rows.
    #[arg(long, value_name = "N")]
    pub limit: Option<u64>,
    /// Replace an existing output file.
    #[arg(long)]
    pub overwrite: bool,
    /// What to do with an unparseable row: fail or skip.
    #[arg(long, default_value = "fail", value_name = "POLICY")]
    pub on_error: String,
    /// Report path (default: `<output>.report.json`).
    #[arg(long, value_name = "PATH")]
    pub report: Option<PathBuf>,
    /// Do not write a report sidecar.
    #[arg(long)]
    pub no_report: bool,
    /// Recurse into subdirectories when the input is a directory.
    #[arg(long)]
    pub recursive: bool,
    /// Suppress progress output.
    #[arg(short, long)]
    pub quiet: bool,
}

/// Parse a byte size as plain bytes (``1073741824``) or with a suffix
/// (``1GiB``, ``1024MiB``, ``512MB``).
pub fn parse_size(value: &str) -> Result<u64, String> {
    let text = value.trim();
    let split = text
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(text.len());
    let (number, suffix) = text.split_at(split);
    let number: f64 = number
        .parse()
        .map_err(|_| format!("invalid size {value:?}; expected 1073741824 or 1GiB"))?;
    let multiplier = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" => 1000.0,
        "kib" => 1024.0,
        "m" | "mb" => 1_000_000.0,
        "mib" => 1024.0 * 1024.0,
        "g" | "gb" => 1_000_000_000.0,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        other => {
            return Err(format!(
                "unknown size suffix {other:?}; use B, KiB, MiB, GiB, KB, MB or GB"
            ))
        }
    };
    if !(number > 0.0) || !number.is_finite() {
        return Err(format!("size {value:?} must be positive"));
    }
    Ok((number * multiplier) as u64)
}

#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Parquet file to summarize.
    #[arg(value_name = "PATH")]
    pub path: PathBuf,
    /// Emit the summary as JSON.
    #[arg(long)]
    pub json: bool,
    /// List the first row group row counts only.
    #[arg(long)]
    pub brief: bool,
}

#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// Source JSONL file.
    #[arg(value_name = "SOURCE")]
    pub source: PathBuf,
    /// Compiled parquet file.
    #[arg(value_name = "COMPILED")]
    pub compiled: PathBuf,
    /// Rows to compare, starting at --offset.
    #[arg(long, default_value_t = 1000, value_name = "N")]
    pub rows: u64,
    /// Source rows to skip before comparing; use the tail of a long file.
    #[arg(long, default_value_t = 0, value_name = "N")]
    pub offset: u64,
    /// Corpus family: auto, pretrain, sft, dpo, agent or generic.
    #[arg(long, default_value = "auto", value_name = "FORMAT")]
    pub format: String,
    /// Column holding pretrain text.
    #[arg(long, default_value = "text", value_name = "NAME")]
    pub text_key: String,
    /// Ignore a row-count difference (still compares the sample).
    #[arg(long)]
    pub allow_row_mismatch: bool,
}
