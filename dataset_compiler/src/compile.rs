//! One-file compilation: schema pre-scan, then conversion and writing.
//!
//! The schema is derived from a full sequential scan by default.  A parquet
//! file has exactly one schema, so a key discovered late must be modelled
//! before the first record batch is written, and only a full scan can promise
//! that.  ``--pre-scan off`` trades that guarantee for one less read on
//! corpora whose rows are known to be uniform.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use rayon::prelude::*;
use rayon::ThreadPool;
use serde_json::Value;

use crate::build::{build_batch, convert_row, Cell};
use crate::reader::{spawn_reader, ChunkStream, LineChunk, ReaderStats};
use crate::report::{
    iso8601_now, normalize, ColumnSummary, InputSummary, OutputSummary, Report,
};
use crate::schema::{detect_preset, Preset, SchemaBuilder, SchemaPlan};
use crate::write::{ParquetSink, SinkStats, WriterOptions};

/// Rows per record batch unless overridden.  Small enough that a correction
/// stays cheap, large enough that per-batch Arrow allocation disappears.
pub const DEFAULT_BATCH_ROWS: usize = 2048;
/// Rows sampled when ``--pre-scan off``.
pub const DEFAULT_SCHEMA_ROWS: usize = 8192;
/// Raw source bytes buffered per chunk.  Bounds memory for corpora whose rows
/// are megabytes long, where a row count alone would not.
const CHUNK_BYTES: usize = 32 * 1024 * 1024;
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Footer keys the streaming planner reads back.  Kept here as the single
/// definition of the contract between the compiler and
/// ``dataset/streaming_chunks.py``.
pub const ALIGNED_CHUNK_BYTES_KEY: &str = "instinct.aligned_chunk_bytes";
pub const ALIGNED_CHUNK_ROWS_KEY: &str = "instinct.aligned_chunk_rows";

/// Reproduces the JSONL streaming planner's byte-driven chunk boundaries, so a
/// compiled file can be cut into one row group per chunk and a training cursor
/// keeps its meaning across containers.
///
/// The rule is the one in ``build_jsonl_chunk_plan``: a chunk closes *before*
/// the first line that starts at or past ``chunk_start + chunk_bytes``, and the
/// next boundary is measured from that line.
#[derive(Debug, Clone)]
struct Alignment {
    chunk_bytes: u64,
    boundaries: Vec<u64>,
    rows: u64,
    next_boundary: u64,
}

impl Alignment {
    fn new(chunk_bytes: u64) -> Alignment {
        Alignment {
            chunk_bytes,
            boundaries: Vec::new(),
            rows: 0,
            next_boundary: chunk_bytes,
        }
    }

    /// One data row, in file order, with the byte offset where it starts.
    fn observe(&mut self, line_start: u64) {
        if self.rows > 0 && line_start >= self.next_boundary {
            self.boundaries.push(self.rows);
            self.next_boundary = line_start + self.chunk_bytes;
        }
        self.rows += 1;
    }

    /// Cumulative row count at the end of every chunk; the last entry is the
    /// total, matching the planner's chunk list.
    fn finish(mut self) -> Vec<u64> {
        self.boundaries.push(self.rows);
        debug_assert!(
            self.boundaries.windows(2).all(|pair| pair[0] < pair[1]),
            "chunk boundaries must strictly increase"
        );
        self.boundaries
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnError {
    Fail,
    Skip,
}

impl OnError {
    pub fn parse(value: &str) -> Result<OnError> {
        match value.to_ascii_lowercase().as_str() {
            "fail" | "error" => Ok(OnError::Fail),
            "skip" | "ignore" => Ok(OnError::Skip),
            other => bail!("unknown on-error policy {other:?}; expected fail or skip"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Explicit preset; only consulted when ``auto_detect`` is off.
    pub format: Preset,
    pub auto_detect: bool,
    pub text_key: String,
    pub writer: WriterOptions,
    pub batch_rows: usize,
    pub threads: usize,
    pub pre_scan: bool,
    pub schema_rows: usize,
    pub limit: Option<u64>,
    pub overwrite: bool,
    pub on_error: OnError,
    pub quiet: bool,
    pub progress: bool,
    /// ``--align-chunk-bytes``: cut row groups at the same rows the JSONL
    /// streaming planner cuts chunks of this many bytes.
    pub align_chunk_bytes: Option<u64>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            format: Preset::Generic,
            auto_detect: true,
            text_key: "text".to_string(),
            writer: WriterOptions::default(),
            batch_rows: DEFAULT_BATCH_ROWS,
            threads: 0,
            pre_scan: true,
            schema_rows: DEFAULT_SCHEMA_ROWS,
            limit: None,
            overwrite: false,
            on_error: OnError::Fail,
            quiet: false,
            progress: true,
            align_chunk_bytes: None,
        }
    }
}

/// Rows that could not be parsed, with a bounded sample of the reasons.
#[derive(Default)]
struct ErrorTally {
    count: AtomicU64,
    samples: Mutex<Vec<String>>,
}

impl ErrorTally {
    fn record(&self, message: String) {
        self.count.fetch_add(1, Ordering::Relaxed);
        let mut samples = self
            .samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if samples.len() < 5 {
            samples.push(message);
        }
    }

    fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    fn samples(&self) -> Vec<String> {
        self.samples
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

/// A row that is parsed but not yet converted, because the schema it must fit
/// is still being inferred.
enum Pending {
    Value(Value),
    Failed,
}

pub fn compile_file(input: &Path, output: &Path, options: &CompileOptions) -> Result<Report> {
    if !input.is_file() {
        bail!("input is not a file: {}", input.display());
    }
    if output.exists() && !options.overwrite {
        bail!(
            "{} already exists; pass --overwrite to replace it",
            output.display()
        );
    }
    let started = Instant::now();
    let threads = resolve_threads(options.threads);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("dataset-compiler-{index}"))
        .build()
        .context("cannot start the parse pool")?;
    let input_summary = InputSummary::read(input)?;
    let batch_rows = options.batch_rows.max(1);
    let errors = ErrorTally::default();

    let preset = if options.auto_detect {
        detect_preset(&first_row(input)?, &options.text_key)
    } else {
        options.format
    };

    // The pre-scan reads the file once and the write pass reads it again, so
    // each pass gets its own reader.  Sampling shares one reader with the write
    // pass, because it stops inside the first window it will write.
    let mut boundaries: Vec<u64> = Vec::new();
    let (plan, buffered, mut stream) = if options.pre_scan {
        // The scan sees every row the write pass will see, so its tally would
        // double count.  It exists only to make `--on-error fail` fatal early.
        let scan_errors = ErrorTally::default();
        let (plan, scan_boundaries, scan_stats) = scan_schema(
            &pool,
            input,
            preset,
            options,
            &scan_errors,
            threads,
            batch_rows,
        )?;
        if options.align_chunk_bytes.is_some() && scan_stats.blank_rows > 0 {
            // The JSONL planner counts a blank line as a row while the compiler
            // does not, so byte-driven boundaries could not be reproduced.
            bail!(
                "--align-chunk-bytes needs a corpus without blank lines; {} has {}",
                input.display(),
                scan_stats.blank_rows
            );
        }
        if let Some(chunk_bytes) = options.align_chunk_bytes {
            let total = *scan_boundaries.last().unwrap_or(&0);
            if total != plan.rows {
                bail!(
                    "aligned boundaries cover {total} rows but the scan saw {}",
                    plan.rows
                );
            }
            if !options.quiet {
                eprintln!(
                    "[dataset_compiler] aligned to {} chunks of <= {} MiB of source bytes",
                    scan_boundaries.len(),
                    chunk_bytes / (1024 * 1024)
                );
            }
        }
        boundaries = scan_boundaries;
        let stream = spawn_reader(input, batch_rows, CHUNK_BYTES, threads.max(1), options.limit)?;
        (plan, Vec::new(), stream)
    } else {
        if options.align_chunk_bytes.is_some() {
            bail!("--align-chunk-bytes requires the full schema scan (--pre-scan true)");
        }
        let mut stream =
            spawn_reader(input, batch_rows, CHUNK_BYTES, threads.max(1), options.limit)?;
        let (plan, buffered) = sample_schema(&pool, &mut stream, preset, options, &errors)?;
        (plan, buffered, stream)
    };
    if plan.columns.is_empty() {
        bail!("{} has no usable rows", input.display());
    }
    if !options.quiet {
        eprintln!(
            "[dataset_compiler] format={} columns={} schema_rows={} elapsed={:.1}s",
            plan.preset.name(),
            plan.columns.len(),
            plan.rows,
            started.elapsed().as_secs_f64()
        );
    }

    let metadata = alignment_metadata(options, &boundaries);
    let mut sink = ParquetSink::create(output, plan.arrow_schema(), &options.writer, &metadata)?;
    let mut last_report = Instant::now();
    let mut boundary_index = 0usize;
    let mut file_row = 0u64;
    for batch in buffered.chunks(batch_rows) {
        sink.write(&build_batch(&plan, batch)?)?;
        file_row += batch.len() as u64;
    }
    while let Some(chunk) = stream.next_chunk() {
        let cells = parse_chunk(&pool, &plan, &chunk, options, &errors)?;
        stream.recycle(chunk);
        // An aligned chunk boundary can fall inside this read chunk, so the
        // batch is split there and the row group closed.
        let mut consumed = 0usize;
        while consumed < cells.len() {
            let take = next_segment(
                &boundaries,
                boundary_index,
                file_row + consumed as u64,
                cells.len() - consumed,
            );
            sink.write(&build_batch(&plan, &cells[consumed..consumed + take])?)?;
            consumed += take;
            if boundaries.get(boundary_index) == Some(&(file_row + consumed as u64)) {
                sink.flush_row_group()?;
                boundary_index += 1;
            }
        }
        file_row += cells.len() as u64;
        if options.progress && last_report.elapsed() >= PROGRESS_INTERVAL {
            eprintln!(
                "[dataset_compiler] rows={} written={:.1} MiB elapsed={:.1}s",
                sink.rows_written(),
                sink.bytes_written() as f64 / (1024.0 * 1024.0),
                started.elapsed().as_secs_f64()
            );
            last_report = Instant::now();
        }
    }
    // Close the trailing row group; a flush with nothing buffered is a no-op.
    sink.flush_row_group()?;
    let reader_stats = stream.finish()?;
    let stats = sink.finish()?;
    if reader_stats.rows != stats.rows {
        bail!(
            "compiled {} rows but read {}; the output is incomplete",
            stats.rows,
            reader_stats.rows
        );
    }
    Ok(build_report(
        output,
        &plan,
        &stats,
        input_summary,
        reader_stats,
        errors,
        options,
        threads,
        started.elapsed(),
        &boundaries,
    ))
}

/// Parse the first non-blank row; ``--format auto`` names the family from it.
fn first_row(input: &Path) -> Result<Value> {
    crate::reader::first_row(input)
}

fn resolve_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(4)
}

/// Full sequential scan: the schema covers every row of the source, and the
/// aligned chunk boundaries are derived from the same pass.
fn scan_schema(
    pool: &ThreadPool,
    input: &Path,
    preset: Preset,
    options: &CompileOptions,
    errors: &ErrorTally,
    threads: usize,
    batch_rows: usize,
) -> Result<(SchemaPlan, Vec<u64>, ReaderStats)> {
    let mut stream = spawn_reader(input, batch_rows, CHUNK_BYTES, threads.max(1), options.limit)?;
    let mut merged = SchemaBuilder::new(preset, &options.text_key);
    let mut alignment = options.align_chunk_bytes.map(Alignment::new);
    let pieces = pool.current_num_threads().max(1);
    while let Some(chunk) = stream.next_chunk() {
        if let Some(alignment) = alignment.as_mut() {
            for index in 0..chunk.len() {
                alignment.observe(chunk.start(index));
            }
        }
        let ranges = split_ranges(chunk.len(), pieces);
        let parts: Vec<Result<SchemaBuilder>> = pool.install(|| {
            ranges
                .par_iter()
                .map(|(start, end)| {
                    let mut builder = SchemaBuilder::new(preset, &options.text_key);
                    for index in *start..*end {
                        match parse_line(&chunk, index) {
                            Ok(value) => builder.observe(&value)?,
                            Err(error) => report_row_error(error, options, errors)?,
                        }
                    }
                    Ok(builder)
                })
                .collect()
        });
        for part in parts {
            merged.merge(&part?);
        }
        stream.recycle(chunk);
    }
    let stats = stream.finish()?;
    let boundaries = alignment.map(Alignment::finish).unwrap_or_default();
    Ok((merged.finish()?, boundaries, stats))
}

/// Sample leading rows instead of scanning the file twice.  Every row read
/// here is converted and returned, so no row is dropped when the schema locks.
fn sample_schema(
    pool: &ThreadPool,
    stream: &mut ChunkStream,
    preset: Preset,
    options: &CompileOptions,
    errors: &ErrorTally,
) -> Result<(SchemaPlan, Vec<Vec<Cell>>)> {
    let target = options.schema_rows.max(1);
    let mut builder = SchemaBuilder::new(preset, &options.text_key);
    let mut pending: Vec<Pending> = Vec::new();
    while let Some(chunk) = stream.next_chunk() {
        let parsed: Vec<Result<Value>> = pool.install(|| {
            (0..chunk.len())
                .into_par_iter()
                .map(|index| parse_line(&chunk, index))
                .collect()
        });
        stream.recycle(chunk);
        for value in parsed {
            match value {
                Ok(value) => {
                    builder.observe(&value)?;
                    pending.push(Pending::Value(value));
                }
                Err(error) => {
                    report_row_error(error, options, errors)?;
                    pending.push(Pending::Failed);
                }
            }
        }
        if pending.len() >= target {
            break;
        }
    }
    let plan = builder.finish()?;
    let placeholder = vec![Cell::Null; plan.columns.len()];
    let rows = pending
        .into_iter()
        .map(|item| match item {
            Pending::Value(value) => convert_row(&value, &plan),
            Pending::Failed => Ok(placeholder.clone()),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((plan, rows))
}

fn parse_chunk(
    pool: &ThreadPool,
    plan: &SchemaPlan,
    chunk: &LineChunk,
    options: &CompileOptions,
    errors: &ErrorTally,
) -> Result<Vec<Vec<Cell>>> {
    pool.install(|| {
        (0..chunk.len())
            .into_par_iter()
            .map(|index| -> Result<Vec<Cell>> {
                match parse_line(chunk, index).and_then(|value| convert_row(&value, plan)) {
                    Ok(cells) => Ok(cells),
                    Err(error) => {
                        report_row_error(error, options, errors)?;
                        Ok(vec![Cell::Null; plan.columns.len()])
                    }
                }
            })
            .collect()
    })
}

fn parse_line(chunk: &LineChunk, index: usize) -> Result<Value> {
    serde_json::from_slice::<Value>(chunk.line(index))
        .map_err(|error| anyhow!("row {}: invalid JSON: {error}", chunk.row_number(index)))
}

fn report_row_error(
    error: anyhow::Error,
    options: &CompileOptions,
    errors: &ErrorTally,
) -> Result<()> {
    if options.on_error == OnError::Fail {
        return Err(error);
    }
    errors.record(error.to_string());
    Ok(())
}

fn split_ranges(count: usize, pieces: usize) -> Vec<(usize, usize)> {
    if count == 0 {
        return Vec::new();
    }
    let slice = count.div_ceil(pieces.max(1)).max(1);
    (0..count)
        .step_by(slice)
        .map(|start| (start, (start + slice).min(count)))
        .collect()
}

/// Rows to write before the next aligned boundary closes a row group.
fn next_segment(boundaries: &[u64], index: usize, absolute_start: u64, remaining: usize) -> usize {
    match boundaries.get(index) {
        Some(&boundary) if boundary > absolute_start => {
            ((boundary - absolute_start) as usize).min(remaining)
        }
        _ => remaining,
    }
}

/// Footer metadata that lets the streaming planner reuse these exact chunks.
fn alignment_metadata(options: &CompileOptions, boundaries: &[u64]) -> Vec<(String, String)> {
    match options.align_chunk_bytes {
        Some(chunk_bytes) if !boundaries.is_empty() => vec![
            (ALIGNED_CHUNK_BYTES_KEY.to_string(), chunk_bytes.to_string()),
            (
                ALIGNED_CHUNK_ROWS_KEY.to_string(),
                boundaries
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ),
        ],
        _ => Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    output: &Path,
    plan: &SchemaPlan,
    stats: &SinkStats,
    input_summary: InputSummary,
    reader_stats: ReaderStats,
    errors: ErrorTally,
    options: &CompileOptions,
    threads: usize,
    elapsed: Duration,
    boundaries: &[u64],
) -> Report {
    let mut warnings: Vec<String> = plan.notes.clone();
    for sample in errors.samples() {
        warnings.push(format!("skipped {sample}"));
    }
    if errors.count() > 0 {
        warnings.push(format!(
            "{} rows could not be parsed and were written as all-null placeholders",
            errors.count()
        ));
    }
    let seconds = elapsed.as_secs_f64();
    Report {
        name: output
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_default(),
        created_at: iso8601_now(),
        tool: format!("dataset_compiler {}", env!("CARGO_PKG_VERSION")),
        format: plan.preset.name().to_string(),
        input: input_summary,
        output: OutputSummary {
            path: normalize(output),
            bytes: stats.bytes,
            rows: stats.rows,
            row_groups: stats.row_groups,
            sha256: stats.sha256.clone(),
        },
        compression: options.writer.compression_label(),
        row_group_rows: options.writer.row_group_rows,
        batch_rows: options.batch_rows,
        threads,
        schema_scanned_rows: plan.rows,
        aligned_chunk_bytes: options.align_chunk_bytes.filter(|_| !boundaries.is_empty()),
        aligned_chunks: (!boundaries.is_empty()).then_some(boundaries.len()),
        skipped_rows: errors.count(),
        elapsed_seconds: seconds,
        throughput_mb_per_s: if seconds > 0.0 {
            reader_stats.bytes as f64 / (1024.0 * 1024.0) / seconds
        } else {
            0.0
        },
        columns: plan
            .describe()
            .into_iter()
            .map(|(name, data_type, nullable)| ColumnSummary {
                name,
                data_type,
                nullable,
            })
            .collect(),
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::CompressionArg;
    use std::io::Write;

    fn write_jsonl(lines: &[&str]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
        file.flush().unwrap();
        file
    }

    fn compile(lines: &[&str], options: CompileOptions) -> (tempfile::TempDir, Report) {
        let input = write_jsonl(lines);
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("out.parquet");
        let report = compile_file(input.path(), &output, &options).unwrap();
        (directory, report)
    }

    fn quiet_options() -> CompileOptions {
        CompileOptions {
            quiet: true,
            progress: false,
            threads: 2,
            ..CompileOptions::default()
        }
    }

    #[test]
    fn text_corpus_compiles_with_auto_detection() {
        let (_dir, report) = compile(
            &[
                "{\"text\": \"hello\", \"token_count\": 2}",
                "{\"text\": \"world\"}",
            ],
            quiet_options(),
        );
        assert_eq!(report.format, "pretrain");
        assert_eq!(report.output.rows, 2);
        assert_eq!(report.output.row_groups, 1);
        assert_eq!(report.columns[0].name, "text");
        assert_eq!(report.columns[1].name, "token_count");
        assert_eq!(report.skipped_rows, 0);
        assert_eq!(report.output.sha256.len(), 64);
    }

    #[test]
    fn chat_corpus_compiles_with_chosen_format() {
        let options = CompileOptions {
            format: Preset::Sft,
            auto_detect: false,
            ..quiet_options()
        };
        let (_dir, report) = compile(
            &[
                "{\"conversations\": [{\"role\": \"user\", \"content\": \"hi\"}]}",
                "{\"conversations\": [{\"role\": \"user\", \"content\": \"yo\"}]}",
            ],
            options,
        );
        assert_eq!(report.format, "sft");
        assert_eq!(report.columns[0].data_type, "list<struct<role, content>>");
    }

    #[test]
    fn skip_policy_keeps_row_counts_aligned() {
        let options = CompileOptions {
            on_error: OnError::Skip,
            ..quiet_options()
        };
        let (_dir, report) = compile(
            &["{\"text\": \"ok\"}", "{not json}", "{\"text\": \"ok again\"}"],
            options,
        );
        assert_eq!(report.output.rows, 3);
        assert_eq!(report.skipped_rows, 1);
        assert!(report.warnings.iter().any(|note| note.contains("all-null")));
    }

    #[test]
    fn fail_policy_reports_the_offending_row() {
        let input = write_jsonl(&["{\"text\": \"ok\"}", "{not json}"]);
        let directory = tempfile::tempdir().unwrap();
        let error = compile_file(
            input.path(),
            &directory.path().join("out.parquet"),
            &quiet_options(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("row 2"), "unexpected error: {error}");
    }

    #[test]
    fn existing_output_requires_overwrite() {
        let input = write_jsonl(&["{\"text\": \"ok\"}"]);
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("out.parquet");
        std::fs::write(&output, b"occupied").unwrap();
        assert!(compile_file(input.path(), &output, &quiet_options()).is_err());
        let options = CompileOptions {
            overwrite: true,
            ..quiet_options()
        };
        assert!(compile_file(input.path(), &output, &options).is_ok());
    }

    #[test]
    fn sampled_schema_keeps_every_row() {
        let lines = [
            "{\"text\": \"one\"}",
            "{\"text\": \"two\", \"token_count\": 1}",
            "{\"text\": \"three\"}",
        ];
        let scanned = CompileOptions {
            pre_scan: true,
            ..quiet_options()
        };
        let sampled = CompileOptions {
            pre_scan: false,
            schema_rows: 2,
            ..quiet_options()
        };
        let (_, first) = compile(&lines, scanned);
        let (_, second) = compile(&lines, sampled);
        assert_eq!(first.output.rows, 3);
        assert_eq!(second.output.rows, 3);
        assert_eq!(
            first
                .columns
                .iter()
                .map(|column| column.name.clone())
                .collect::<Vec<_>>(),
            second
                .columns
                .iter()
                .map(|column| column.name.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn sampled_schema_locks_inside_a_large_first_chunk() {
        // One chunk holds every row here, so the schema must lock without the
        // rows that arrived with it being lost.
        let lines: Vec<String> = (0..200)
            .map(|index| format!("{{\"text\": \"row {index}\"}}"))
            .collect();
        let borrowed: Vec<&str> = lines.iter().map(String::as_str).collect();
        let options = CompileOptions {
            pre_scan: false,
            schema_rows: 4,
            ..quiet_options()
        };
        let (_dir, report) = compile(&borrowed, options);
        assert_eq!(report.output.rows, 200);
    }

    #[test]
    fn limit_stops_at_the_requested_row() {
        let options = CompileOptions {
            limit: Some(2),
            ..quiet_options()
        };
        let (_dir, report) = compile(
            &["{\"text\": \"a\"}", "{\"text\": \"b\"}", "{\"text\": \"c\"}"],
            options,
        );
        assert_eq!(report.output.rows, 2);
    }

    #[test]
    fn compression_choice_is_recorded() {
        let options = CompileOptions {
            writer: WriterOptions {
                compression: CompressionArg::Zstd,
                compression_level: Some(3),
                ..WriterOptions::default()
            },
            ..quiet_options()
        };
        let (_dir, report) = compile(&["{\"text\": \"a\"}"], options);
        assert_eq!(report.compression, "zstd(3)");
    }

    #[test]
    fn empty_input_is_rejected() {
        let input = write_jsonl(&[]);
        let directory = tempfile::tempdir().unwrap();
        assert!(compile_file(
            input.path(),
            &directory.path().join("out.parquet"),
            &quiet_options()
        )
        .is_err());
    }

    #[test]
    fn row_order_survives_parallel_parsing() {
        let lines: Vec<String> = (0..500)
            .map(|index| format!("{{\"text\": \"row {index}\"}}"))
            .collect();
        let borrowed: Vec<&str> = lines.iter().map(String::as_str).collect();
        let options = CompileOptions { threads: 4, ..quiet_options() };
        let (_dir, report) = compile(&borrowed, options);
        assert_eq!(report.output.rows, 500);
    }

    #[test]
    fn alignment_closes_a_chunk_before_the_crossing_row() {
        // Mirrors ``build_jsonl_chunk_plan``: a chunk closes before the first
        // line that starts at or past ``chunk_start + chunk_bytes``.
        let mut alignment = Alignment::new(10);
        // Byte offsets:   0   6    12   18   24
        for start in [0u64, 6, 12, 18, 24] {
            alignment.observe(start);
        }
        // Row 3 starts at 12 >= 10, so the first chunk ends after row 2.
        assert_eq!(alignment.finish(), vec![2, 4, 5]);
    }

    #[test]
    fn alignment_handles_a_leading_oversized_row() {
        let mut alignment = Alignment::new(10);
        for start in [0u64, 500, 505] {
            alignment.observe(start);
        }
        assert_eq!(alignment.finish(), vec![1, 3]);
    }

    #[test]
    fn aligned_compile_emits_one_row_group_per_chunk() {
        let lines: Vec<String> = (0..7)
            .map(|index| format!("{{\"text\": \"row {index:02}\"}}"))
            .collect();
        let borrowed: Vec<&str> = lines.iter().map(String::as_str).collect();
        // Two rows of source bytes per chunk, whatever a row costs.
        let chunk_bytes = (lines[0].len() as u64 + 1) * 2;
        let options = CompileOptions {
            align_chunk_bytes: Some(chunk_bytes),
            ..quiet_options()
        };
        let (directory, report) = compile(&borrowed, options);

        assert_eq!(report.output.rows, 7);
        assert_eq!(report.aligned_chunk_bytes, Some(chunk_bytes));
        assert_eq!(report.aligned_chunks, Some(4));
        assert_eq!(report.output.row_groups, 4);

        let summary = crate::read::summarize(&directory.path().join("out.parquet")).unwrap();
        assert_eq!(summary.row_group_rows, vec![2, 2, 2, 1]);
        let metadata = crate::read::footer_metadata(&directory.path().join("out.parquet")).unwrap();
        assert_eq!(
            metadata.get(ALIGNED_CHUNK_ROWS_KEY).map(String::as_str),
            Some("2,4,6,7")
        );
        assert_eq!(
            metadata.get(ALIGNED_CHUNK_BYTES_KEY).map(String::as_str),
            Some(chunk_bytes.to_string().as_str())
        );
    }

    #[test]
    fn unaligned_compile_writes_no_alignment_metadata() {
        let (_dir, report) = compile(&["{\"text\": \"a\"}"], quiet_options());
        assert!(report.aligned_chunk_bytes.is_none());
        assert!(report.aligned_chunks.is_none());
    }

    #[test]
    fn alignment_requires_the_pre_scan() {
        let options = CompileOptions {
            pre_scan: false,
            align_chunk_bytes: Some(1024),
            ..quiet_options()
        };
        let input = write_jsonl(&["{\"text\": \"a\"}"]);
        let directory = tempfile::tempdir().unwrap();
        let error = compile_file(input.path(), &directory.path().join("o.parquet"), &options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("full schema scan"), "unexpected error: {error}");
    }

    #[test]
    fn alignment_refuses_a_corpus_with_blank_lines() {
        let input = write_jsonl(&["{\"text\": \"a\"}", "", "{\"text\": \"b\"}"]);
        let directory = tempfile::tempdir().unwrap();
        let options = CompileOptions {
            align_chunk_bytes: Some(8),
            ..quiet_options()
        };
        let error = compile_file(input.path(), &directory.path().join("o.parquet"), &options)
            .unwrap_err()
            .to_string();
        assert!(error.contains("blank lines"), "unexpected error: {error}");
    }

    #[test]
    fn size_suffixes_parse() {
        use crate::cli::parse_size;

        assert_eq!(parse_size("1073741824").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size("1GiB").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size("1024MiB").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_size("2gib").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("512MB").unwrap(), 512_000_000);
        assert!(parse_size("").is_err());
        assert!(parse_size("0GiB").is_err());
        assert!(parse_size("10TiB").is_err());
    }
}
