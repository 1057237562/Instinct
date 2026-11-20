//! Copy encoded column chunks without decoding/recompressing the corpus.
//! Each shard has its own writer; Rayon runs bounded concurrent shard jobs.
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use arrow_array::{Array, Int64Array};
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ProjectionMask};
use parquet::column::writer::ColumnCloseResult;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::cli::{SplitArgs, VerifyShardsArgs};
use crate::compile::{ALIGNED_CHUNK_BYTES_KEY, ALIGNED_CHUNK_ROWS_KEY};

#[derive(Debug, Serialize)]
struct Shard {
    file: String,
    source_row_group_start: usize,
    source_row_group_end_exclusive: usize,
    source_row_start: i64,
    source_row_end_exclusive: i64,
    rows: i64,
    bytes: u64,
    token_count: Option<i64>,
    sha256: String,
    encoded_columns_verified: usize,
    elapsed_seconds: f64,
}

fn plan(sizes: &[u64], target: u64) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut size: u64 = 0;
    for (index, bytes) in sizes.iter().copied().enumerate() {
        if index > start && size.abs_diff(target) <= (size + bytes).abs_diff(target) {
            ranges.push(start..index);
            start = index;
            size = 0;
        }
        size += bytes;
    }
    if start < sizes.len() {
        ranges.push(start..sizes.len());
    }
    ranges
}

pub fn run(args: &SplitArgs) -> Result<()> {
    ensure!(
        args.threads > 0 && args.target_size > 0,
        "threads and target size must be positive"
    );
    let source = fs::canonicalize(&args.input)?;
    let before = fs::metadata(&source)?;
    let identity = source_identity(&source)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&source)?)?;
    let metadata = builder.metadata().clone();
    ensure!(
        metadata.num_row_groups() > 0,
        "cannot split an empty corpus"
    );
    // External column files cannot be copied safely as standalone shards.
    for group in metadata.row_groups() {
        for column in group.columns() {
            ensure!(
                column.file_path().is_none(),
                "external column files are unsupported"
            );
        }
    }
    let sizes: Vec<_> = metadata
        .row_groups()
        .iter()
        .map(|g| g.columns().iter().map(|c| c.compressed_size() as u64).sum())
        .collect();
    let ranges = plan(&sizes, args.target_size);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    // create_dir is deliberately exclusive: never overwrite existing data.
    fs::create_dir(&args.output)
        .context("output directory must be new and its parent must exist")?;
    let started = Instant::now();
    eprintln!(
        "split: {} rows, {} row groups -> {} shards, {} workers",
        metadata.file_metadata().num_rows(),
        sizes.len(),
        ranges.len(),
        args.threads
    );
    let shards: Vec<Shard> = pool.install(|| {
        ranges
            .par_iter()
            .enumerate()
            .map(|(index, range)| {
                let name = format!("part-{:05}-of-{:05}.parquet", index + 1, ranges.len());
                let shard = copy_shard(&source, &args.output, &name, &metadata, range.clone())?;
                eprintln!(
                    "verified {}: {} rows, {} bytes, tokens={:?}, {:.1}s",
                    name, shard.rows, shard.bytes, shard.token_count, shard.elapsed_seconds
                );
                Ok(shard)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let after = fs::metadata(&source)?;
    ensure!(
        before.len() == after.len() && before.modified()? == after.modified()?,
        "source changed during split"
    );
    let rows: i64 = shards.iter().map(|s| s.rows).sum();
    ensure!(
        rows == metadata.file_metadata().num_rows(),
        "row conservation failed"
    );
    let tokens: Option<i64> = shards.iter().map(|s| s.token_count).sum();
    let report = serde_json::json!({
        "status": "complete", "tool": "dataset_compiler split", "source": source,
        "source_bytes": before.len(), "source_rows": rows,
        "source_modified": format!("{:?}", before.modified()?),
        "source_identity": identity,
        "source_row_group_rows": metadata.row_groups().iter().map(|g| g.num_rows()).collect::<Vec<_>>(),
        "target_bytes": args.target_size, "threads": args.threads,
        "token_count": tokens, "row_groups": sizes.len(),
        "verification": "SHA-256 of every encoded column chunk compared with source; schema, rows and token sums checked per shard",
        "order": "contiguous source row groups; concatenate filenames in lexical order",
        "resume_note": "A manifest-aware native loader can retain the original source identity and global row-group cursor across these verified shards.",
        "elapsed_seconds": started.elapsed().as_secs_f64(), "shards": shards,
    });
    let report_tmp = args.output.join("manifest.json.partial");
    fs::write(&report_tmp, serde_json::to_vec_pretty(&report)?)?;
    fs::rename(report_tmp, args.output.join("manifest.json"))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub fn source_identity(path: &Path) -> Result<serde_json::Value> {
    let file = File::open(path)?;
    let stat = file.metadata()?;
    let mut input = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0; stat.len().min(65536) as usize];
    input.read_exact(&mut buffer)?;
    digest.update(&buffer);
    if stat.len() > 65536 {
        input.seek(SeekFrom::End(-65536))?;
        buffer.resize(65536, 0);
        input.read_exact(&mut buffer)?;
        digest.update(&buffer);
    }
    Ok(
        serde_json::json!({"path":crate::report::normalize(path), "bytes":stat.len(),
        "mtime_ns":stat.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos(),
        "sample_sha256":format!("{:x}", digest.finalize())}),
    )
}

pub fn verify_command(args: &VerifyShardsArgs) -> Result<()> {
    println!(
        "{}",
        verify_shards(&args.manifest, args.threads, args.file.as_deref())?
    );
    Ok(())
}

fn verify_shards(path: &Path, threads: usize, selected: Option<&str>) -> Result<serde_json::Value> {
    ensure!(threads > 0, "threads must be positive");
    let root = fs::canonicalize(path.parent().context("manifest needs a parent")?)?;
    let report: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(report["status"] == "complete", "split is incomplete");
    ensure!(
        report["source_identity"].is_object(),
        "manifest has no original source identity"
    );
    let rows = report["source_row_group_rows"]
        .as_array()
        .context("missing original group rows")?;
    let shards = report["shards"].as_array().context("missing shards")?;
    ensure!(!shards.is_empty(), "empty shards");
    ensure!(
        selected.is_none() || shards.iter().any(|s| s["file"].as_str() == selected),
        "selected shard is not in manifest"
    );
    let mut next_group = 0usize;
    let mut next_row = 0u64;
    let mut files = Vec::new();
    for shard in shards {
        let name = shard["file"].as_str().context("invalid shard filename")?;
        ensure!(
            Path::new(name).components().count() == 1,
            "shard must be a direct child"
        );
        let active = selected.is_none() || selected == Some(name);
        let file = if active {
            fs::canonicalize(root.join(name))?
        } else {
            root.join(name)
        };
        ensure!(
            file.parent() == Some(root.as_path()) && !files.contains(&file),
            "invalid/duplicate shard path"
        );
        let end = shard["source_row_group_end_exclusive"]
            .as_u64()
            .context("missing group end")? as usize;
        ensure!(
            shard["source_row_group_start"].as_u64() == Some(next_group as u64)
                && end > next_group
                && end <= rows.len(),
            "gap/overlap in original row groups"
        );
        ensure!(
            shard["source_row_start"].as_u64() == Some(next_row),
            "gap/overlap in original rows"
        );
        let count: u64 = rows[next_group..end]
            .iter()
            .map(|v| v.as_u64().context("invalid group rows"))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum();
        if active {
            let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&file)?)?;
            ensure!(
                builder.metadata().num_row_groups() == end - next_group,
                "shard group count mismatch"
            );
            for (local, expected) in rows[next_group..end].iter().enumerate() {
                ensure!(
                    expected.as_i64() == Some(builder.metadata().row_group(local).num_rows()),
                    "shard group rows mismatch"
                );
            }
            ensure!(
                builder.metadata().file_metadata().num_rows() as u64 == count,
                "shard row count mismatch"
            );
            ensure!(
                fs::metadata(&file)?.len()
                    == shard["bytes"].as_u64().context("missing shard size")?,
                "shard size mismatch"
            );
        }
        next_row += count;
        ensure!(
            shard["rows"].as_u64() == Some(count)
                && shard["source_row_end_exclusive"].as_u64() == Some(next_row),
            "shard row range mismatch"
        );
        next_group = end;
        files.push(file);
    }
    ensure!(
        next_group == rows.len()
            && report["row_groups"].as_u64() == Some(next_group as u64)
            && report["source_rows"].as_u64() == Some(next_row),
        "incomplete source coverage"
    );
    let identities: Vec<_> = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?
        .install(|| {
            files
                .par_iter()
                .zip(shards.par_iter())
                .map(|(file, shard)| {
                    if selected.is_some() && selected != shard["file"].as_str() {
                        return Ok(serde_json::Value::Null);
                    }
                    let before = source_identity(file)?;
                    let bytes = fs::metadata(file)?.len();
                    ensure!(
                        Some(digest_range(file, 0, bytes)?.as_str()) == shard["sha256"].as_str(),
                        "shard SHA-256 mismatch: {}",
                        file.display()
                    );
                    ensure!(
                        before == source_identity(file)?,
                        "shard changed during verification"
                    );
                    Ok(before)
                })
                .collect::<Result<Vec<_>>>()
        })?;
    Ok(serde_json::json!({"verified":true,"files":identities,"manifest":report}))
}

fn copy_shard(
    source: &Path,
    output: &Path,
    name: &str,
    metadata: &ParquetMetaData,
    range: Range<usize>,
) -> Result<Shard> {
    let started = Instant::now();
    let file_metadata = metadata.file_metadata();
    let mut kv = file_metadata
        .key_value_metadata()
        .cloned()
        .unwrap_or_default();
    // Aligned boundaries are cumulative row ends, local to this new file.
    if kv.iter().any(|v| v.key == ALIGNED_CHUNK_BYTES_KEY) {
        let mut rows = 0i64;
        let boundaries: Vec<_> = range
            .clone()
            .map(|g| {
                rows += metadata.row_group(g).num_rows();
                rows
            })
            .collect();
        let value = serde_json::to_string(&boundaries)?;
        if let Some(entry) = kv.iter_mut().find(|v| v.key == ALIGNED_CHUNK_ROWS_KEY) {
            entry.value = Some(value);
        } else {
            bail!("aligned source is missing chunk boundaries");
        }
    }
    let properties = WriterProperties::builder()
        .set_created_by(
            file_metadata
                .created_by()
                .unwrap_or("dataset_compiler split")
                .to_owned(),
        )
        .set_key_value_metadata(Some(kv))
        .build();
    let partial = output.join(format!("{name}.partial"));
    let destination = output.join(name);
    let input = File::open(source)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)?;
    let mut writer = SerializedFileWriter::new(
        BufWriter::with_capacity(1024 * 1024, file),
        file_metadata.schema_descr_ptr().root_schema_ptr(),
        Arc::new(properties),
    )?;
    for index in range.clone() {
        let group = metadata.row_group(index);
        let mut sink = writer.next_row_group()?;
        for column in group.columns() {
            sink.append_column(
                &input,
                ColumnCloseResult {
                    bytes_written: column.compressed_size() as u64,
                    rows_written: group.num_rows() as u64,
                    metadata: column.clone(),
                    bloom_filter: None,
                    column_index: None,
                    offset_index: None,
                },
            )?;
        }
        sink.close()?;
    }
    writer.close()?;
    OpenOptions::new().write(true).open(&partial)?.sync_all()?;
    let result = ParquetRecordBatchReaderBuilder::try_new(File::open(&partial)?)?;
    let expected_rows: i64 = range
        .clone()
        .map(|i| metadata.row_group(i).num_rows())
        .sum();
    ensure!(
        result.metadata().file_metadata().num_rows() == expected_rows,
        "shard row count mismatch"
    );
    ensure!(
        result.metadata().num_row_groups() == range.len(),
        "shard row-group count mismatch"
    );
    ensure!(
        result.metadata().file_metadata().schema_descr() == file_metadata.schema_descr(),
        "shard schema mismatch"
    );
    let mut verified = 0;
    for (local, original) in range.clone().enumerate() {
        let expected = metadata.row_group(original);
        let actual = result.metadata().row_group(local);
        ensure!(
            expected.num_rows() == actual.num_rows(),
            "row group mismatch"
        );
        for (left, right) in expected.columns().iter().zip(actual.columns()) {
            ensure!(
                left.compressed_size() == right.compressed_size()
                    && left.num_values() == right.num_values(),
                "column metadata mismatch"
            );
            let (lo, ll) = left.byte_range();
            let (ro, rl) = right.byte_range();
            ensure!(
                digest_range(source, lo, ll)? == digest_range(&partial, ro, rl)?,
                "encoded column SHA-256 mismatch"
            );
            verified += 1;
        }
    }
    let source_tokens = token_sum(source, range.clone())?;
    ensure!(
        source_tokens == token_sum(&partial, 0..range.len())?,
        "token conservation failed"
    );
    let bytes = fs::metadata(&partial)?.len();
    let sha256 = digest_range(&partial, 0, bytes)?;
    let row_start: i64 = (0..range.start)
        .map(|i| metadata.row_group(i).num_rows())
        .sum();
    fs::rename(&partial, destination)?;
    Ok(Shard {
        file: name.to_owned(),
        source_row_group_start: range.start,
        source_row_group_end_exclusive: range.end,
        source_row_start: row_start,
        source_row_end_exclusive: row_start + expected_rows,
        rows: expected_rows,
        bytes,
        token_count: source_tokens,
        sha256,
        encoded_columns_verified: verified,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn digest_range(path: &Path, offset: u64, length: u64) -> Result<String> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut input = BufReader::with_capacity(1024 * 1024, file.take(length));
    let mut hasher = Sha256::new();
    let mut block = vec![0; 1024 * 1024];
    let mut count = 0u64;
    loop {
        let read = input.read(&mut block)?;
        if read == 0 {
            break;
        }
        hasher.update(&block[..read]);
        count += read as u64;
    }
    ensure!(count == length, "truncated column data");
    Ok(format!("{:x}", hasher.finalize()))
}

fn token_sum(path: &Path, groups: Range<usize>) -> Result<Option<i64>> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let Ok(index) = builder.schema().index_of("token_count") else {
        return Ok(None);
    };
    let mask = ProjectionMask::roots(builder.parquet_schema(), [index]);
    let reader = builder
        .with_projection(mask)
        .with_row_groups(groups.collect())
        .with_batch_size(8192)
        .build()?;
    let mut sum = 0i64;
    for batch in reader {
        let batch = batch?;
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("token_count must be int64")?;
        for index in 0..values.len() {
            if !values.is_null(index) {
                sum = sum
                    .checked_add(values.value(index))
                    .context("token sum overflow")?;
            }
        }
    }
    Ok(Some(sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile_file, CompileOptions};
    use crate::read::read_values_iter;

    #[test]
    fn nearest_group_boundaries_and_oversized_groups() {
        assert_eq!(plan(&[38, 38, 38, 38, 38], 76), vec![0..2, 2..4, 4..5]);
        assert_eq!(plan(&[200, 10, 10], 50), vec![0..1, 1..3]);
    }

    #[test]
    fn split_preserves_every_value_and_rebases_alignment() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let source = dir.path().join("source.jsonl");
        fs::write(&source, "{\"text\":\"alpha\",\"token_count\":3,\"extra\":{\"x\":1}}\n{\"text\":\"beta\",\"token_count\":5}\n{\"text\":\"gamma\",\"token_count\":7}\n")?;
        let parquet = dir.path().join("source.parquet");
        let options = CompileOptions {
            align_chunk_bytes: Some(1),
            ..Default::default()
        };
        compile_file(&source, &parquet, &options)?;
        let output = dir.path().join("shards");
        let args = SplitArgs {
            input: parquet.clone(),
            output: output.clone(),
            target_size: 1,
            threads: 2,
        };
        run(&args)?;
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("manifest.json"))?)?;
        assert_eq!(report["source_rows"], 3);
        assert_eq!(report["token_count"], 15);
        let original: Vec<_> = read_values_iter(&parquet, 0, None)?
            .1
            .collect::<Result<_>>()?;
        let mut actual = Vec::new();
        for shard in report["shards"].as_array().unwrap() {
            let path = output.join(shard["file"].as_str().unwrap());
            actual.extend(
                read_values_iter(&path, 0, None)?
                    .1
                    .collect::<Result<Vec<_>>>()?,
            );
            let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
            let kv = builder
                .metadata()
                .file_metadata()
                .key_value_metadata()
                .unwrap();
            let alignment = kv.iter().find(|v| v.key == ALIGNED_CHUNK_ROWS_KEY).unwrap();
            assert_eq!(alignment.value.as_deref(), Some("[1]"));
        }
        assert_eq!(actual, original);
        assert_eq!(report["source_identity"], source_identity(&parquet)?);
        assert_eq!(
            verify_shards(&output.join("manifest.json"), 2, None)?["verified"],
            true
        );
        let first = output.join(report["shards"][0]["file"].as_str().unwrap());
        let mut bytes = fs::read(&first)?;
        bytes[4] ^= 1;
        fs::write(first, bytes)?;
        assert!(verify_shards(&output.join("manifest.json"), 2, None).is_err());
        let second = report["shards"][1]["file"].as_str().unwrap();
        assert_eq!(
            verify_shards(&output.join("manifest.json"), 2, Some(second))?["verified"],
            true
        );
        assert!(run(&args).is_err());
        Ok(())
    }
}
