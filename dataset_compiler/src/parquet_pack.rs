//! Bounded native preprocessing for the Torch-free cuBLAS trainer.
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};
use arrow_array::{Array, Int32Array, Int64Array, LargeStringArray, StringArray,
                  StringViewArray, UInt32Array, UInt64Array};
use clap::Args;
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ProjectionMask};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokenizers::{Tokenizer, TruncationParams};

#[derive(Debug, Args)]
pub struct PlanArgs {
    #[arg(long)] pub input: PathBuf,
    #[arg(long, default_value_t = 4096)] pub max_length: usize,
    #[arg(long, default_value_t = 4)] pub threads: usize,
}

#[derive(Debug, Args)]
pub struct PackArgs {
    #[arg(long)] pub input: PathBuf,
    #[arg(long)] pub output: PathBuf,
    #[arg(long)] pub tokenizer: PathBuf,
    #[arg(long)] pub row_group: usize,
    #[arg(long, value_delimiter = ',', default_value = "3072,4096")]
    pub lengths: Vec<usize>,
    #[arg(long, default_value_t = 1000)] pub packing_batch_size: usize,
    #[arg(long, default_value_t = 4)] pub threads: usize,
    #[arg(long, default_value_t = 42)] pub seed: u64,
    #[arg(long, default_value_t = 1)] pub bos: u16,
    #[arg(long, default_value_t = 2)] pub eos: u16,
    #[arg(long, default_value_t = 0)] pub pad: u16,
    #[arg(long, default_value = "")] pub fingerprint: String,
}

#[derive(Serialize, PartialEq, Eq)]
struct Source {
    path: String, bytes: u64, mtime_ns: u128, sample_sha256: String,
}

fn source(path: &Path) -> Result<Source> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 65536];
    let count = file.read(&mut buffer)?;
    digest.update(&buffer[..count]);
    if metadata.len() > 65536 {
        file.seek(SeekFrom::End(-65536))?;
        file.read_exact(&mut buffer)?;
        digest.update(&buffer);
    }
    Ok(Source {path: path.to_string_lossy().into_owned(), bytes: metadata.len(),
        mtime_ns: metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos(),
        sample_sha256: format!("{:x}", digest.finalize())})
}

fn pool(threads: usize) -> Result<rayon::ThreadPool> {
    ensure!(threads > 0, "threads must be positive");
    Ok(rayon::ThreadPoolBuilder::new().num_threads(threads).build()?)
}

fn token_count(array: &dyn Array, index: usize) -> Result<u64> {
    ensure!(!array.is_null(index), "null token_count at row {index}");
    macro_rules! signed {
        ($ty:ty) => { if let Some(values) = array.as_any().downcast_ref::<$ty>() {
            return Ok(u64::try_from(values.value(index)).context("negative token_count")?);
        }};
    }
    signed!(Int32Array); signed!(Int64Array); signed!(UInt32Array); signed!(UInt64Array);
    bail!("token_count must be an integer column")
}

#[derive(Serialize)]
struct Group { index: usize, row_start: u64, rows: u64, tokens: u64 }

pub fn plan_command(args: PlanArgs) -> Result<()> {
    ensure!(args.max_length >= 2, "max length must be at least two");
    let identity = source(&args.input)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(&args.input)?)?;
    let sizes: Vec<_> = builder.metadata().row_groups().iter().map(|g| g.num_rows() as u64).collect();
    let counts: Result<Vec<u64>> = pool(args.threads)?.install(|| (0..sizes.len()).into_par_iter().map(|group| {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&args.input)?)?;
        let index = reader.schema().index_of("token_count").context("production Parquet requires token_count metadata")?;
        let projection = ProjectionMask::roots(reader.parquet_schema(), [index]);
        let mut count = 0;
        for batch in reader.with_projection(projection).with_row_groups(vec![group]).with_batch_size(16384).build()? {
            let batch = batch?;
            for row in 0..batch.num_rows() {
                count += token_count(batch.column(0).as_ref(), row)?.min(args.max_length as u64);
            }
        }
        Ok(count)
    }).collect());
    let mut rows = 0;
    let groups: Vec<_> = sizes.into_iter().zip(counts?).enumerate().map(|(index, (size, tokens))| {
        let group = Group {index, row_start: rows, rows: size, tokens}; rows += size; group
    }).collect();
    ensure!(source(&args.input)? == identity, "source changed during planning");
    println!("{}", serde_json::json!({"format":1,"source":identity,"rows":rows,
        "tokens":groups.iter().map(|g|g.tokens).sum::<u64>(),"groups":groups}));
    Ok(())
}

fn text(array: &dyn Array, row: usize) -> Result<&str> {
    ensure!(!array.is_null(row), "null pretraining text at row {row}");
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() { return Ok(values.value(row)); }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() { return Ok(values.value(row)); }
    if let Some(values) = array.as_any().downcast_ref::<StringViewArray>() { return Ok(values.value(row)); }
    bail!("pretraining text must be an Arrow string column")
}

// Stable, versioned Fisher-Yates permutation; no dependency on a library RNG's defaults.
fn shuffle(values: &mut [usize], mut state: u64) {
    for end in (1..values.len()).rev() {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        values.swap(end, (z % (end as u64 + 1)) as usize);
    }
}

fn best_fit(documents: &[Vec<u16>], indices: &[usize], capacity: usize) -> Vec<Vec<usize>> {
    let mut ordered = indices.to_vec();
    ordered.sort_by_key(|&i| std::cmp::Reverse(documents[i].len()));
    let mut free: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut bins: Vec<Vec<usize>> = Vec::new();
    for index in ordered {
        let size = documents[index].len();
        let available = free.range(size..).next().map(|(&space, _)| space);
        let (bin, space) = if let Some(space) = available {
            let values = free.get_mut(&space).unwrap();
            let bin = values.pop().unwrap();
            if values.is_empty() { free.remove(&space); }
            (bin, space)
        } else { bins.push(Vec::new()); (bins.len() - 1, capacity) };
        bins[bin].push(index);
        if space > size { free.entry(space - size).or_default().push(bin); }
    }
    bins
}

#[derive(Serialize)]
struct Bucket {
    length: usize, blocks: usize, source_rows: usize, valid_tokens: u64, useful_targets: u64,
    tokens_file: String, offsets_file: String, lengths_file: String,
    tokens_bytes: u64, offsets_bytes: u64, lengths_bytes: u64,
}

fn write_bucket(root: &Path, documents: &[Vec<u16>], mut indices: Vec<usize>,
                length: usize, args: &PackArgs) -> Result<Bucket> {
    shuffle(&mut indices, args.seed.wrapping_add(length as u64));
    let packed: Vec<_> = indices.par_chunks(args.packing_batch_size)
        .map(|chunk| best_fit(documents, chunk, length)).collect();
    let tokens_file = format!("{length}.tokens.u16");
    let offsets_file = format!("{length}.offsets.u64");
    let lengths_file = format!("{length}.lengths.u16");
    let mut output = BufWriter::with_capacity(1 << 20, File::create(root.join(&tokens_file))?);
    let mut offsets = vec![0_u64];
    let mut lengths = Vec::<u16>::new();
    let mut buffer = Vec::<u16>::with_capacity(length);
    let (mut valid, mut useful) = (0_u64, 0_u64);
    for batch in packed {
        for bin in batch {
            buffer.clear();
            for index in bin {
                let document = &documents[index];
                valid += document.len() as u64;
                useful += document[1..].iter().filter(|&&token| token != args.pad).count() as u64;
                buffer.extend_from_slice(document);
                lengths.push(u16::try_from(document.len())?);
            }
            ensure!(buffer.len() <= length, "packing overflow");
            buffer.resize(length, args.pad);
            output.write_all(bytemuck::cast_slice(&buffer))?;
            offsets.push(lengths.len() as u64);
        }
    }
    output.flush()?; output.get_ref().sync_all()?;
    fs::write(root.join(&offsets_file), bytemuck::cast_slice(&offsets))?;
    fs::write(root.join(&lengths_file), bytemuck::cast_slice(&lengths))?;
    let blocks = offsets.len() - 1;
    Ok(Bucket {length, blocks, source_rows: indices.len(), valid_tokens: valid, useful_targets: useful,
        tokens_bytes: blocks as u64 * length as u64 * 2, offsets_bytes: offsets.len() as u64 * 8,
        lengths_bytes: lengths.len() as u64 * 2, tokens_file, offsets_file, lengths_file})
}

pub fn pack_command(mut args: PackArgs) -> Result<()> {
    ensure!(cfg!(target_endian = "little"), "cache format requires a little-endian writer");
    args.lengths.sort_unstable(); args.lengths.dedup();
    ensure!(!args.lengths.is_empty() && args.packing_batch_size > 0, "empty packing configuration");
    ensure!(args.lengths.iter().all(|&x| (2..=65535).contains(&x)), "bucket lengths outside uint16 range");
    ensure!(!args.output.exists(), "refusing to overwrite existing cache {}", args.output.display());
    let started = Instant::now();
    let identity = source(&args.input)?;
    let pool = pool(args.threads)?;
    let mut tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    tokenizer.with_padding(None);
    tokenizer.with_truncation(Some(TruncationParams {max_length: args.lengths.last().unwrap() - 2,
        ..Default::default()})).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&args.input)?)?;
    ensure!(args.row_group < reader.metadata().num_row_groups(), "row group outside source");
    let expected_rows = reader.metadata().row_group(args.row_group).num_rows() as usize;
    let row_start: u64 = reader.metadata().row_groups()[..args.row_group].iter().map(|g| g.num_rows() as u64).sum();
    let text_index = reader.schema().index_of("text")?;
    let projection = ProjectionMask::roots(reader.parquet_schema(), [text_index]);
    let mut documents = Vec::with_capacity(expected_rows);
    for batch in reader.with_projection(projection).with_row_groups(vec![args.row_group]).with_batch_size(4096).build()? {
        let batch = batch?;
        let strings: Result<Vec<_>> = (0..batch.num_rows()).map(|i| text(batch.column(0).as_ref(), i)).collect();
        let encoded: Result<Vec<Vec<u16>>> = pool.install(|| strings?.par_iter().map(|s| {
            let encoding = tokenizer.encode(*s, false).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            let mut result = Vec::with_capacity(encoding.len() + 2); result.push(args.bos);
            for &token in encoding.get_ids() { result.push(u16::try_from(token).context("token ID exceeds uint16")?); }
            result.push(args.eos); Ok(result)
        }).collect());
        documents.extend(encoded?);
    }
    ensure!(documents.len() == expected_rows, "source row conservation failed");
    let tokenize_seconds = started.elapsed().as_secs_f64();
    eprintln!("[Native Parquet] group={} rows={} tokenization={:.2}s threads={}", args.row_group, documents.len(), tokenize_seconds, args.threads);
    let parent = args.output.parent().context("cache output needs a parent")?;
    fs::create_dir_all(parent)?;
    let work = parent.join(format!(".build-{}-{}", args.output.file_name().unwrap().to_string_lossy(), std::process::id()));
    fs::create_dir(&work)?;
    let mut assigned = vec![Vec::new(); args.lengths.len()];
    for (index, document) in documents.iter().enumerate() {
        let bucket = args.lengths.partition_point(|&length| length < document.len());
        ensure!(bucket < assigned.len(), "document exceeds largest bucket");
        assigned[bucket].push(index);
    }
    let buckets: Result<Vec<_>> = pool.install(|| assigned.into_par_iter().zip(args.lengths.par_iter())
        .map(|(indices, &length)| write_bucket(&work, &documents, indices, length, &args)).collect());
    let buckets = buckets?;
    ensure!(buckets.iter().map(|b| b.source_rows).sum::<usize>() == expected_rows, "packed row conservation failed");
    ensure!(source(&args.input)? == identity, "source changed during preprocessing");
    let manifest = serde_json::json!({"format":1,"fingerprint":args.fingerprint,"source":identity,
        "row_group":args.row_group,"row_start":row_start,"rows":expected_rows,"buckets":buckets,
        "valid_tokens":buckets.iter().map(|b|b.valid_tokens).sum::<u64>(),
        "useful_targets":buckets.iter().map(|b|b.useful_targets).sum::<u64>(),
        "tokenize_seconds":tokenize_seconds,"total_seconds":started.elapsed().as_secs_f64(),
        "threads":args.threads,"seed":args.seed,"packing_batch_size":args.packing_batch_size});
    fs::write(work.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    for attempt in 0..8 {
        match fs::rename(&work, &args.output) {
            Ok(()) => { println!("{}", manifest); return Ok(()); }
            Err(error) if attempt < 7 && error.kind() == std::io::ErrorKind::PermissionDenied => {
                std::thread::sleep(Duration::from_millis(50 * (1 << attempt.min(3))));
            }
            Err(error) => return Err(error).with_context(|| format!("publish failed; completed cache retained in {}", work.display())),
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn best_fit_preserves_every_complete_document() {
        let documents: Vec<_> = [9, 2, 7, 3, 5, 6, 1, 8].iter().enumerate().map(|(i, &n)| vec![i as u16; n]).collect();
        let mut indices: Vec<_> = (0..documents.len()).collect();
        shuffle(&mut indices, 42);
        let bins = best_fit(&documents, &indices, 10);
        assert!(bins.iter().all(|b| b.iter().map(|&i| documents[i].len()).sum::<usize>() <= 10));
        let mut restored: Vec<_> = bins.into_iter().flatten().collect(); restored.sort_unstable();
        assert_eq!(restored, (0..documents.len()).collect::<Vec<_>>());
        let mut again: Vec<_> = (0..documents.len()).collect(); shuffle(&mut again, 42);
        assert_eq!(indices, again);
    }
}
