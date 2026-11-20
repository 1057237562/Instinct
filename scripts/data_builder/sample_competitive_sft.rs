//! Sample competitive SFT candidates by question, language and serialized length.
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread;

const WINDOW: usize = 640;
const BUFFER: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    priority: u64,
    offset: u64,
    bytes: usize,
}

struct Index {
    source: PathBuf,
    held_out: HashSet<u128>,
    selected: HashMap<(u128, u8), Vec<Candidate>>,
    scanned: u64,
}

fn field<'a>(tail: &'a [u8], name: &[u8]) -> Option<&'a [u8]> {
    let position = tail.windows(name.len()).rposition(|part| part == name)? + name.len();
    Some(tail[position..].trim_ascii_start())
}

fn question_id(tail: &[u8]) -> Option<u128> {
    let value = field(tail, b"\"question_id\":")?.strip_prefix(b"\"")?;
    if value.get(32) != Some(&b'"') {
        return None;
    }
    u128::from_str_radix(std::str::from_utf8(&value[..32]).ok()?, 16).ok()
}

fn is_train(tail: &[u8]) -> bool {
    let Some(split) = field(tail, b"\"split\":") else { return false };
    if split.starts_with(b"\"train\"") {
        return true;
    }
    let Some(list) = split.strip_prefix(b"[") else { return false };
    let Some(end) = list.iter().position(|byte| *byte == b']') else { return false };
    list[..end].split(|byte| *byte == b',').all(|value| value.trim_ascii() == b"\"train\"")
}

fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn index(source: PathBuf) -> io::Result<Index> {
    let mut reader = BufReader::with_capacity(BUFFER, File::open(&source)?);
    let mut index = Index { source, held_out: HashSet::new(), selected: HashMap::new(), scanned: 0 };
    let (mut line, mut offset) = (Vec::new(), 0);
    loop {
        line.clear();
        let bytes = reader.read_until(b'\n', &mut line)?;
        if bytes == 0 { break; }
        index.scanned += 1;
        let tail = &line[bytes.saturating_sub(WINDOW)..];
        let id = question_id(tail).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData,
            format!("{} row {} lacks a question ID", index.source.display(), index.scanned)))?;
        if !is_train(tail) {
            index.held_out.insert(id);
        } else if bytes <= 61_000 {
            let band = if bytes <= 14_000 { 0 } else if bytes <= 31_000 { 1 } else { 2 };
            let row = Candidate { priority: mix(offset ^ id as u64 ^ 20261113), offset, bytes };
            let top = index.selected.entry((id, band)).or_default();
            if top.len() < 3 {
                top.push(row);
            } else if let Some(worst) = top.iter_mut().max() {
                if row < *worst { *worst = row; }
            }
        }
        offset += bytes as u64;
        if index.scanned % 200_000 == 0 {
            eprintln!("[index] {}: {} rows, {} bytes", index.source.display(), index.scanned, offset);
        }
    }
    Ok(index)
}

fn write_sample(index: Index, held_out: &HashSet<u128>, directory: &Path) -> io::Result<()> {
    let stem = index.source.file_stem().unwrap().to_string_lossy();
    let path = directory.join(format!("{stem}.tsv"));
    let temporary = path.with_extension("partial");
    let mut reader = File::open(&index.source)?;
    let mut output = BufWriter::with_capacity(BUFFER, File::create(&temporary)?);
    let mut rows: Vec<_> = index.selected.into_iter().filter(|((id, _), _)| !held_out.contains(id))
        .flat_map(|(_, rows)| rows).collect();
    rows.sort_unstable_by_key(|row| row.offset);
    let mut line = Vec::new();
    for row in &rows {
        line.resize(row.bytes, 0);
        reader.seek(SeekFrom::Start(row.offset))?;
        reader.read_exact(&mut line)?;
        write!(output, "{}\t", row.offset)?;
        output.write_all(&line)?;
    }
    output.flush()?;
    drop(output);
    std::fs::rename(temporary, path)?;
    println!("{stem}: scanned {}, sampled {}, holdout IDs {}", index.scanned, rows.len(), held_out.len());
    Ok(())
}

fn main() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let directory = PathBuf::from(args.next().ok_or_else(|| io::Error::other("missing output directory"))?);
    let paths: Vec<_> = args.map(PathBuf::from).collect();
    if paths.len() != 4 {
        return Err(io::Error::other("expected all four raw shards to exclude held-out question IDs"));
    }
    std::fs::create_dir_all(&directory)?;
    let indexed = thread::scope(|scope| {
        let tasks: Vec<_> = paths.into_iter().map(|path| scope.spawn(move || index(path))).collect();
        tasks.into_iter().map(|task| task.join().expect("index worker panicked")).collect::<io::Result<Vec<_>>>()
    })?;
    let held_out: HashSet<_> = indexed.iter().flat_map(|index| index.held_out.iter().copied()).collect();
    thread::scope(|scope| {
        let tasks: Vec<_> = indexed.into_iter().map(|index| {
            let (held_out, directory) = (&held_out, directory.as_path());
            scope.spawn(move || write_sample(index, held_out, directory))
        }).collect();
        tasks.into_iter().try_for_each(|task| task.join().expect("sample writer panicked"))
    })
}
