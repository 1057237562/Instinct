//! `dataset_compiler`: turn Instinct JSONL corpora into Parquet.

mod build;
mod cli;
mod compile;
mod read;
mod reader;
mod report;
mod schema;
mod verify;
mod write;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;

use crate::cli::{Cli, Command, CompileArgs, InspectArgs, VerifyArgs};
use crate::compile::{compile_file, CompileOptions, DEFAULT_BATCH_ROWS, DEFAULT_SCHEMA_ROWS};
use crate::report::Report;
use crate::schema::Preset;
use crate::write::{CompressionArg, DictionaryArg, WriterOptions};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Compile(args) => compile_command(args),
        Command::Inspect(args) => inspect_command(args),
        Command::Verify(args) => verify_command(args),
    }
}

fn compile_command(args: CompileArgs) -> Result<()> {
    let options = compile_options(&args)?;
    if args.input.is_dir() {
        return compile_directory(&args, &options);
    }
    if !args.input.exists() {
        bail!("input does not exist: {}", args.input.display());
    }
    if args.output.is_dir() {
        bail!(
            "--output {} is a directory but --input is a file",
            args.output.display()
        );
    }
    let started = std::time::Instant::now();
    let report = compile_file(&args.input, &args.output, &options)?;
    if let Some(path) = report_path(&args) {
        report.write(&path)?;
    }
    print_report(&report, &args, started.elapsed().as_secs_f64());
    Ok(())
}

/// Compile every top-level ``*.jsonl`` of a directory into it or beside it.
fn compile_directory(args: &CompileArgs, options: &CompileOptions) -> Result<()> {
    if args.report.is_some() {
        bail!("--report names one file; write reports beside each output instead");
    }
    let inputs = jsonl_files(&args.input, args.recursive)?;
    if inputs.is_empty() {
        bail!("no .jsonl files in {}", args.input.display());
    }
    if args.output.extension().is_some_and(|value| value == "parquet") && inputs.len() > 1 {
        bail!("--output must be a directory when the input holds several files");
    }
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("cannot create {}", args.output.display()))?;
    let total = inputs.len();
    let mut failures = Vec::new();
    for input in inputs {
        let stem = input
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_else(|| "output".to_string());
        let output = args.output.join(format!("{stem}.parquet"));
        let started = std::time::Instant::now();
        match compile_file(&input, &output, options) {
            Ok(report) => {
                if let Some(path) = report_path_for(&output, args) {
                    report.write(&path)?;
                }
                print_report(&report, args, started.elapsed().as_secs_f64());
            }
            Err(error) => {
                eprintln!("error: {}: {error:#}", input.display());
                failures.push(input);
            }
        }
    }
    if !failures.is_empty() {
        bail!("{}/{total} files failed to compile", failures.len());
    }
    Ok(())
}

fn jsonl_files(directory: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)
            .with_context(|| format!("cannot list {}", current.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                if recursive {
                    stack.push(path);
                }
            } else if path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))
            {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn compile_options(args: &CompileArgs) -> Result<CompileOptions> {
    let (format, auto_detect) = match args.format.to_ascii_lowercase().as_str() {
        "auto" => (Preset::Generic, true),
        "generic" => (Preset::Generic, false),
        other => (Preset::parse(other)?, false),
    };
    if args.row_group_rows == 0 {
        bail!("--row-group-rows must be at least 1");
    }
    if args.row_group_mb == 0 {
        bail!("--row-group-mb must be at least 1");
    }
    Ok(CompileOptions {
        format,
        auto_detect,
        text_key: args.text_key.clone(),
        writer: WriterOptions {
            compression: CompressionArg::parse(&args.compression)?,
            compression_level: args.compression_level,
            row_group_rows: args.row_group_rows,
            row_group_bytes: args.row_group_mb * 1024 * 1024,
            dictionary: DictionaryArg::parse(&args.dictionary)?,
            statistics: args.statistics,
            aligned_chunk_bytes: args.align_chunk_bytes,
        },
        batch_rows: args.batch_rows.max(1),
        threads: args.threads,
        pre_scan: args.pre_scan,
        schema_rows: args.schema_rows.max(1),
        limit: args.limit,
        overwrite: args.overwrite,
        on_error: crate::compile::OnError::parse(&args.on_error)?,
        quiet: args.quiet,
        progress: !args.quiet,
        align_chunk_bytes: args.align_chunk_bytes,
    })
}

fn report_path(args: &CompileArgs) -> Option<PathBuf> {
    report_path_for(&args.output, args)
}

fn report_path_for(output: &Path, args: &CompileArgs) -> Option<PathBuf> {
    if args.no_report {
        return None;
    }
    Some(args.report.clone().unwrap_or_else(|| {
        let mut name = output.as_os_str().to_os_string();
        name.push(".report.json");
        PathBuf::from(name)
    }))
}

fn print_report(report: &Report, args: &CompileArgs, seconds: f64) {
    if args.quiet {
        return;
    }
    let megabytes = report.input.bytes as f64 / (1024.0 * 1024.0);
    println!(
        "{}\n  rows={} rows/s={:.0}\n  source={:.1} MiB  parquet={:.1} MiB  ratio={:.2}x\n  \
         compression={} row_groups={} format={}\n  elapsed={:.1}s ({:.1} MiB/s)  output={}",
        report.name,
        report.output.rows,
        report.output.rows as f64 / seconds.max(1e-9),
        megabytes,
        report.output.bytes as f64 / (1024.0 * 1024.0),
        if report.output.bytes > 0 {
            report.input.bytes as f64 / report.output.bytes as f64
        } else {
            0.0
        },
        report.compression,
        report.output.row_groups,
        report.format,
        seconds,
        report.throughput_mb_per_s,
        report.output.path,
    );
    if !report.warnings.is_empty() {
        for warning in &report.warnings {
            println!("  warning: {warning}");
        }
    }
    if let (Some(chunk_bytes), Some(chunks)) = (report.aligned_chunk_bytes, report.aligned_chunks) {
        println!(
            "  aligned: {chunks} chunks, one row group each, cut at the same bytes the \
             streaming planner uses for {} MiB of JSONL",
            chunk_bytes / (1024 * 1024)
        );
    }
}

fn inspect_command(args: InspectArgs) -> Result<()> {
    let summary = read::summarize(&args.path)?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary).context("cannot render the summary")?
        );
        return Ok(());
    }
    println!("{}", summary.path);
    println!(
        "  rows={} row_groups={} bytes={:.2} GiB",
        summary.rows,
        summary.row_groups,
        summary.bytes as f64 / GIB
    );
    if let Some(created_by) = &summary.created_by {
        println!("  created_by={created_by}");
    }
    if !args.brief {
        println!("  arrow schema: {}", summary.arrow_schema);
        let rows = summary
            .row_group_rows
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        println!("  row_group_rows=[{rows}]");
    }
    Ok(())
}

fn verify_command(args: VerifyArgs) -> Result<()> {
    let (format, auto_detect) = match args.format.to_ascii_lowercase().as_str() {
        "auto" => (Preset::Generic, true),
        "generic" => (Preset::Generic, false),
        other => (Preset::parse(other)?, false),
    };
    let outcome = verify::verify(
        &args.source,
        &args.compiled,
        &verify::VerifyOptions {
            format,
            auto_detect,
            text_key: args.text_key.clone(),
            rows: args.rows,
            offset: args.offset,
            allow_row_mismatch: args.allow_row_mismatch,
        },
    )?;
    println!("{}", outcome.summary());
    for mismatch in &outcome.mismatches {
        println!("  {mismatch}");
    }
    if !outcome.ok() {
        bail!("verification failed");
    }
    Ok(())
}

/// Kept for the help text and tests: the defaults live with the compile flow.
#[allow(dead_code)]
pub fn defaults() -> (usize, usize) {
    (DEFAULT_BATCH_ROWS, DEFAULT_SCHEMA_ROWS)
}
