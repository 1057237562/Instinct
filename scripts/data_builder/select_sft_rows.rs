//! Copy planned, one-based JSONL rows verbatim, using one bounded reader per source.
//! Build: rustc -O scripts/data_builder/select_sft_rows.rs -o tmp/select_sft_rows.exe
use std::collections::HashSet;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::thread;

fn select(job: &[String]) -> io::Result<()> {
    let [source, indices, target] = job else { unreachable!() };
    let mut wanted = BufReader::new(File::open(indices)?)
        .lines()
        .map(|line| {
            line?.parse::<u64>().map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
        })
        .collect::<io::Result<HashSet<_>>>()?;
    let mut input = BufReader::with_capacity(1024 * 1024, File::open(source)?);
    let mut output = BufWriter::with_capacity(1024 * 1024, OpenOptions::new().write(true).create_new(true).open(target)?);
    let mut line = Vec::new();
    let mut row = 0;
    while input.read_until(b'\n', &mut line)? != 0 {
        row += 1;
        if wanted.remove(&row) {
            output.write_all(&line)?;
            if !line.ends_with(b"\n") {
                output.write_all(b"\n")?;
            }
        }
        line.clear();
    }
    if !wanted.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{source}: {} planned rows missing", wanted.len())));
    }
    output.flush()?;
    println!("{}: scanned {row} rows", Path::new(source).display());
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.is_empty() || args.len() % 3 != 0 || args.len() > 48 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "expected 1–16 SOURCE INDEX_FILE OUTPUT triples"));
    }
    thread::scope(|scope| {
        let workers: Vec<_> = args.chunks_exact(3).map(|job| scope.spawn(move || select(job))).collect();
        for worker in workers {
            worker.join().expect("JSONL selector panicked")?;
        }
        Ok(())
    })
}
