//! Parallel native merge of ExternalRandomShuffler's sorted 128-bit-key runs.
//! rustc --edition=2021 -O merge_shuffle_runs.rs -o merge_shuffle_runs.exe
use std::{cmp::Ordering, collections::BinaryHeap, fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf}, thread, time::{Instant, Duration}};

fn retry_file(path: &Path, create: bool) -> io::Result<File> {
    for attempt in 0..100 {
        match if create { File::create(path) } else { File::open(path) } {
            Ok(file) => return Ok(file),
            Err(e) if matches!(e.raw_os_error(), Some(32 | 33)) && attempt < 99 => thread::sleep(Duration::from_millis(100)),
            Err(e) => return Err(io::Error::new(e.kind(), format!("{}: {e}", path.display()))),
        }
    }
    unreachable!()
}

struct Item { key: [u8; 32], line: Vec<u8>, source: usize }
impl PartialEq for Item { fn eq(&self, b: &Self) -> bool { self.key == b.key && self.source == b.source } }
impl Eq for Item {}
impl Ord for Item { fn cmp(&self, b: &Self) -> Ordering { b.key.cmp(&self.key).then(b.source.cmp(&self.source)) } }
impl PartialOrd for Item { fn partial_cmp(&self, b: &Self) -> Option<Ordering> { Some(self.cmp(b)) } }

fn boundary(path: &Path, prefix: usize) -> io::Result<u64> {
    let size = path.metadata()?.len();
    if prefix == 0 { return Ok(0); }
    if prefix == 16 { return Ok(size); }
    let mut target = [b'0'; 32];
    target[0] = b"0123456789abcdef"[prefix];
    let mut reader = BufReader::with_capacity(65536, retry_file(path, false)?);
    let (mut low, mut high) = (0, size);
    let mut line = Vec::new();
    while high.saturating_sub(low) > 65536 {
        let mid = low + (high - low) / 2;
        reader.seek(SeekFrom::Start(mid - 1))?;
        let mut previous = [0];
        reader.read_exact(&mut previous)?;
        if previous[0] != b'\n' { line.clear(); reader.read_until(b'\n', &mut line)?; }
        line.clear();
        reader.read_until(b'\n', &mut line)?;
        if line.len() >= 33 && line[..32] < target[..] { low = reader.stream_position()?; }
        else { high = mid; }
    }
    reader.seek(SeekFrom::Start(low))?;
    loop {
        let start = reader.stream_position()?;
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 { return Ok(size); }
        if line.len() < 34 || line[32] != b'\t' { return Err(io::Error::other("Invalid shuffle run")); }
        if line[..32] >= target[..] { return Ok(start); }
    }
}

fn merge_partition(paths: &[PathBuf], prefix: usize, part: &Path) -> io::Result<(u64, u64)> {
    let marker = part.with_extension("done");
    if marker.exists() {
        let text = fs::read_to_string(&marker)?;
        let values: Vec<u64> = text.split_whitespace().map(|s| s.parse().unwrap()).collect();
        if part.metadata()?.len() != values[1] { return Err(io::Error::other("Completed partition size changed")); }
        eprintln!("[reuse partition {prefix:x}] {} rows, {} bytes", values[0], values[1]);
        return Ok((values[0], values[1]));
    }
    let mut streams = Vec::new();
    let mut heap = BinaryHeap::new();
    for (source, path) in paths.iter().enumerate() {
        if prefix == 0 && source % 32 == 0 { eprintln!("[index] {source}/{} runs", paths.len()); }
        let start = boundary(path, prefix)?;
        let end = boundary(path, prefix + 1)?;
        let mut file = retry_file(path, false)?;
        file.seek(SeekFrom::Start(start))?;
        let mut reader = BufReader::with_capacity(256 * 1024, file.take(end - start));
        let mut line = Vec::new();
        if reader.read_until(b'\n', &mut line)? > 0 {
            let key = line[..32].try_into().unwrap();
            heap.push(Item { key, line, source });
        }
        streams.push(reader);
    }
    let mut output = BufWriter::with_capacity(4 * 1024 * 1024, retry_file(part, true)?);
    let (mut rows, mut bytes) = (0, 0);
    while let Some(mut item) = heap.pop() {
        output.write_all(&item.line[33..])?;
        rows += 1;
        bytes += (item.line.len() - 33) as u64;
        item.line.clear();
        if streams[item.source].read_until(b'\n', &mut item.line)? > 0 {
            item.key.copy_from_slice(&item.line[..32]);
            heap.push(item);
        }
    }
    output.flush()?;
    drop(output);
    fs::write(marker, format!("{rows} {bytes}\n"))?;
    eprintln!("[partition {prefix:x}] {rows} rows, {bytes} bytes");
    Ok((rows, bytes))
}

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 5 { return Err(io::Error::other("Usage: merge_shuffle_runs RUN_DIR OUTPUT EXPECTED_ROWS EXPECTED_BYTES")); }
    let run_dir = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    if output.exists() { return Err(io::Error::other("Output exists")); }
    let expected_rows: u64 = args[3].to_str().unwrap().parse().unwrap();
    let expected_bytes: u64 = args[4].to_str().unwrap().parse().unwrap();
    let mut paths: Vec<_> = fs::read_dir(&run_dir)?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("chunk_")).collect();
    paths.sort();
    if paths.is_empty() { return Err(io::Error::other("No sorted runs")); }
    let started = Instant::now();
    let parts: Vec<_> = (0..16).map(|i| run_dir.join(format!("native_part_{i:02}"))).collect();
    let totals = thread::scope(|scope| -> io::Result<(u64, u64)> {
        let handles: Vec<_> = parts.iter().enumerate().map(|(i, part)| {
            let paths = &paths;
            scope.spawn(move || merge_partition(paths, i, part))
        }).collect();
        let (mut rows, mut bytes) = (0, 0);
        for handle in handles { let (r,b) = handle.join().expect("Merge worker panicked")?; rows += r; bytes += b; }
        Ok((rows, bytes))
    })?;
    if totals != (expected_rows, expected_bytes) { return Err(io::Error::other(format!("Conservation failed: {totals:?}"))); }
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, retry_file(&output, true)?);
    for part in &parts { io::copy(&mut retry_file(part, false)?, &mut writer)?; }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    eprintln!("[DONE] {} rows, {} bytes, {:.3}s, 16 native merge workers", totals.0, totals.1, started.elapsed().as_secs_f64());
    Ok(())
}
