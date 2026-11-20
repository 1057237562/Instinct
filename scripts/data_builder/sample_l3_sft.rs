//! Extract bounded, reproducible L3 task/analysis/solution/test samples per shard.
use arrow_array::{Array, LargeStringArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use serde_json::json;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

const FIELDS: [&str; 5] = ["uuid", "task", "analysis", "solution", "test"];
type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn text_at(column: &dyn Array, row: usize) -> Option<&str> {
    if column.is_null(row) {
        return None;
    }
    if let Some(text) = column.as_any().downcast_ref::<StringArray>() {
        return Some(text.value(row));
    }
    column.as_any().downcast_ref::<LargeStringArray>().map(|text| text.value(row))
}

fn sample(source: &str, destination: &Path, shard: usize, limit: usize) -> Result<()> {
    let file = File::open(source)?;
    let footer = ParquetRecordBatchReaderBuilder::try_new(file.try_clone()?)?;
    let fields = FIELDS.map(|name| footer.schema().index_of(name)).into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let groups = footer.metadata().num_row_groups();
    let output = OpenOptions::new().create_new(true).write(true).open(destination)?;
    let mut out = BufWriter::with_capacity(1 << 20, output);
    let spans = groups.min(4);
    let mut written = 0;
    for part in 0..spans {
        let group = (part * groups / spans + shard) % groups;
        let group_rows = footer.metadata().row_group(group).num_rows() as usize;
        let take = (limit / spans).min(group_rows);
        let offset = (1373 * (shard + 1) + 3547 * (part + 1)) % (group_rows - take + 1);
        let reader = ParquetRecordBatchReaderBuilder::try_new(file.try_clone()?)?;
        let projection = ProjectionMask::roots(reader.parquet_schema(), fields.clone());
        let reader = reader.with_row_groups(vec![group]).with_projection(projection)
            .with_offset(offset).with_limit(take).with_batch_size(256).build()?;
        for batch in reader {
            let batch = batch?;
            let columns = FIELDS.map(|name| batch.column_by_name(name).expect("projected field"));
            for row in 0..batch.num_rows() {
                let values = columns.map(|column| text_at(column.as_ref(), row));
                serde_json::to_writer(&mut out, &json!({"source": source, "uuid": values[0],
                    "task": values[1], "analysis": values[2], "solution": values[3], "test": values[4]}))?;
                out.write_all(b"\n")?;
                written += 1;
            }
        }
    }
    out.flush()?;
    println!("{} rows={} file={source}", destination.display(), written);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        return Err("usage: sample_l3_sft OUT_DIR WORKERS ROWS_PER_SHARD SHARD...".into());
    }
    let out = Path::new(&args[0]);
    let workers = args[1].parse::<usize>()?.clamp(1, 16);
    let limit = args[2].parse::<usize>()?;
    std::fs::create_dir(out)?;
    let files = &args[3..];
    std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..workers).map(|worker| scope.spawn(move || -> Result<()> {
            for shard in (worker..files.len()).step_by(workers) {
                sample(&files[shard], &out.join(format!("part-{shard:03}.jsonl")), shard, limit)?;
            }
            Ok(())
        })).collect();
        for job in jobs {
            job.join().expect("sampler worker panicked")?;
        }
        Ok(())
    })
}
