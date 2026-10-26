//! Bounded, parallel JSONL reading.
//!
//! One thread reads whole lines into fixed-size chunks and hands them to a
//! rayon pool, which parses them into schema-aligned cells.  A chunk stores its
//! lines back to back in one byte buffer with an offset per line, so a 37 GiB
//! corpus compiles without per-line allocation and the in-flight window bounds
//! peak memory independently of file size.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::thread::JoinHandle;

use anyhow::{bail, Context, Result};
use crossbeam_channel::{bounded, Receiver, Sender};

/// Read buffer per line; large enough for the longest corpus rows.
const READ_BUFFER_BYTES: usize = 8 * 1024 * 1024;

/// Recycled chunk storage travelling back to the reader thread.
type Reuse = (Vec<u8>, Vec<usize>, Vec<u64>);

/// A contiguous run of raw lines, handed to the parse pool as one unit.
pub struct LineChunk {
    pub first_row: u64,
    /// Line bytes concatenated without terminators.
    data: Vec<u8>,
    /// End offset of each line inside ``data``.
    ends: Vec<usize>,
    /// Absolute offset of each line's first byte in the source file, including
    /// any terminator bytes of earlier lines.  ``--align-chunk-bytes`` needs
    /// this to reproduce byte-driven chunk boundaries exactly.
    starts: Vec<u64>,
}

impl LineChunk {
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn line(&self, index: usize) -> &[u8] {
        let start = if index == 0 { 0 } else { self.ends[index - 1] };
        &self.data[start..self.ends[index]]
    }

    /// Byte offset in the source file where line ``index`` begins.
    pub fn start(&self, index: usize) -> u64 {
        self.starts[index]
    }

    /// Row numbers are 1-based and refer to the source file.
    pub fn row_number(&self, index: usize) -> u64 {
        self.first_row + index as u64 + 1
    }
}

#[derive(Debug, Default)]
pub struct ReaderStats {
    pub bytes: u64,
    pub rows: u64,
    pub blank_rows: u64,
}

pub struct ChunkStream {
    pub chunks: Receiver<LineChunk>,
    recycle: Sender<Reuse>,
    join: JoinHandle<Result<ReaderStats>>,
}

impl ChunkStream {
    pub fn next_chunk(&mut self) -> Option<LineChunk> {
        self.chunks.recv().ok()
    }

    /// Return a consumed chunk's storage to the reader thread.
    pub fn recycle(&self, chunk: LineChunk) {
        let LineChunk {
            mut data,
            mut ends,
            mut starts,
            ..
        } = chunk;
        data.clear();
        ends.clear();
        starts.clear();
        let _ = self.recycle.send((data, ends, starts));
    }

    pub fn finish(self) -> Result<ReaderStats> {
        drop(self.chunks);
        self.join
            .join()
            .map_err(|_| anyhow::anyhow!("the reader thread panicked"))?
    }
}

/// Start the reader thread.  ``inflight`` chunks may be buffered, which sets
/// the memory ceiling at roughly ``inflight * bytes_per_chunk``.
pub fn spawn_reader(
    path: &Path,
    rows_per_chunk: usize,
    bytes_per_chunk: usize,
    inflight: usize,
    limit: Option<u64>,
) -> Result<ChunkStream> {
    let (chunk_tx, chunk_rx) = bounded::<LineChunk>(inflight.max(1));
    let (recycle_tx, recycle_rx) = bounded::<Reuse>(inflight.max(1) + 1);
    let file = File::open(path)
        .with_context(|| format!("cannot open {} for reading", path.display()))?;
    let owned = path.to_path_buf();
    let handle = std::thread::Builder::new()
        .name("dataset-compiler-reader".to_string())
        .spawn(move || -> Result<ReaderStats> {
            read_lines(
                file,
                &owned,
                rows_per_chunk,
                bytes_per_chunk,
                limit,
                &chunk_tx,
                &recycle_rx,
            )
        })
        .context("cannot start the reader thread")?;
    Ok(ChunkStream {
        chunks: chunk_rx,
        recycle: recycle_tx,
        join: handle,
    })
}

fn read_lines(
    file: File,
    path: &Path,
    rows_per_chunk: usize,
    bytes_per_chunk: usize,
    limit: Option<u64>,
    chunks: &Sender<LineChunk>,
    recycle: &Receiver<Reuse>,
) -> Result<ReaderStats> {
    let mut reader = BufReader::with_capacity(READ_BUFFER_BYTES, file);
    let mut stats = ReaderStats::default();
    let mut buffer: Vec<u8> = Vec::with_capacity(64 * 1024);
    let (mut data, mut ends, mut starts) = (Vec::new(), Vec::new(), Vec::new());
    let mut first_row = 0u64;
    let mut first_line = true;
    // Byte offset of the line about to be read, terminators included.
    let mut offset = 0u64;
    loop {
        let line_start = offset;
        buffer.clear();
        let read = reader
            .read_until(b'\n', &mut buffer)
            .with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        stats.bytes += read as u64;
        offset += read as u64;
        // Lines are stored without their terminator, so ``\r\n`` corpora keep
        // the same bytes on either platform.
        while matches!(buffer.last(), Some(b'\n') | Some(b'\r')) {
            buffer.pop();
        }
        if first_line {
            if buffer.starts_with(&[0xEF, 0xBB, 0xBF]) {
                buffer.drain(..3);
            }
            first_line = false;
        }
        if buffer.iter().all(|byte| byte.is_ascii_whitespace()) {
            stats.blank_rows += 1;
            continue;
        }
        data.extend_from_slice(&buffer);
        ends.push(data.len());
        starts.push(line_start);
        stats.rows += 1;
        let reached_limit = limit.is_some_and(|limit| stats.rows >= limit);
        let full = ends.len() >= rows_per_chunk.max(1) || data.len() >= bytes_per_chunk;
        if full || reached_limit {
            let (next_data, next_ends, next_starts) = recycle.try_recv().unwrap_or_default();
            let (payload_data, payload_ends, payload_starts) = (
                std::mem::replace(&mut data, next_data),
                std::mem::replace(&mut ends, next_ends),
                std::mem::replace(&mut starts, next_starts),
            );
            let chunk = LineChunk {
                first_row,
                data: payload_data,
                ends: payload_ends,
                starts: payload_starts,
            };
            first_row += chunk.len() as u64;
            if chunks.send(chunk).is_err() {
                return Ok(stats);
            }
        }
        if reached_limit {
            break;
        }
    }
    if !ends.is_empty() {
        let _ = chunks.send(LineChunk {
            first_row,
            data,
            ends,
            starts,
        });
    }
    Ok(stats)
}

/// Parse the first non-blank line of a file; used to name the corpus family
/// before a full pass runs.  A BOM and a trailing ``\r`` are tolerated.
pub fn first_row(path: &Path) -> Result<serde_json::Value> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut buffer: Vec<u8> = Vec::with_capacity(64 * 1024);
    loop {
        buffer.clear();
        if reader
            .read_until(b'\n', &mut buffer)
            .with_context(|| format!("reading {}", path.display()))?
            == 0
        {
            bail!("{} has no rows", path.display());
        }
        while matches!(buffer.last(), Some(b'\n') | Some(b'\r')) {
            buffer.pop();
        }
        if buffer.starts_with(&[0xEF, 0xBB, 0xBF]) {
            buffer.drain(..3);
        }
        if buffer.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        return serde_json::from_slice(&buffer)
            .map_err(|error| anyhow::anyhow!("row 1: invalid JSON: {error}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn collect(path: &Path, rows_per_chunk: usize) -> (Vec<String>, ReaderStats) {
        let mut stream = spawn_reader(path, rows_per_chunk, usize::MAX, 2, None).unwrap();
        let mut rows = Vec::new();
        while let Some(chunk) = stream.next_chunk() {
            for index in 0..chunk.len() {
                rows.push(String::from_utf8(chunk.line(index).to_vec()).unwrap());
            }
            stream.recycle(chunk);
        }
        let stats = stream.finish().unwrap();
        (rows, stats)
    }

    #[test]
    fn chunks_preserve_order_and_drop_blank_rows() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{\"a\":1}}\n\n{{\"a\":2}}\n{{\"a\":3}}").unwrap();
        file.flush().unwrap();
        let (rows, stats) = collect(file.path(), 2);
        assert_eq!(rows, vec!["{\"a\":1}", "{\"a\":2}", "{\"a\":3}"]);
        assert_eq!(stats.rows, 3);
        assert_eq!(stats.blank_rows, 1);
    }

    #[test]
    fn bom_and_carriage_returns_are_stripped() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all("\u{feff}{\"a\":1}\r\n{\"a\":2}\r\n".as_bytes())
            .unwrap();
        file.flush().unwrap();
        let (rows, _) = collect(file.path(), 16);
        assert_eq!(rows, vec!["{\"a\":1}", "{\"a\":2}"]);
    }

    #[test]
    fn limit_stops_reading_early() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{\"a\":1}}\n{{\"a\":2}}\n{{\"a\":3}}").unwrap();
        file.flush().unwrap();
        let mut stream = spawn_reader(file.path(), 16, usize::MAX, 2, Some(2)).unwrap();
        let mut rows = Vec::new();
        while let Some(chunk) = stream.next_chunk() {
            for index in 0..chunk.len() {
                rows.push(String::from_utf8(chunk.line(index).to_vec()).unwrap());
            }
            stream.recycle(chunk);
        }
        stream.finish().unwrap();
        assert_eq!(rows, vec!["{\"a\":1}", "{\"a\":2}"]);
    }

    #[test]
    fn row_numbers_are_one_based() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{\"a\":1}}\n{{\"a\":2}}\n{{\"a\":3}}").unwrap();
        file.flush().unwrap();
        let mut stream = spawn_reader(file.path(), 2, usize::MAX, 2, None).unwrap();
        let first = stream.next_chunk().unwrap();
        assert_eq!(first.row_number(0), 1);
        assert_eq!(first.row_number(1), 2);
        let second_row = first.first_row + first.len() as u64;
        stream.recycle(first);
        let second = stream.next_chunk().unwrap();
        assert_eq!(second.first_row, second_row);
        assert_eq!(second.row_number(0), 3);
    }
}
