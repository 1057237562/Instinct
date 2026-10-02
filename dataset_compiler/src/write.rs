//! Parquet output with a streaming SHA-256 of the bytes actually written.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use parquet::arrow::ArrowWriter;
use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;
use sha2::{Digest, Sha256};

/// Default page buffer; matches the parquet-rs default but stated explicitly so
/// row-group sizing is predictable.
const WRITE_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionArg {
    None,
    Snappy,
    Zstd,
    Gzip,
    Brotli,
}

impl CompressionArg {
    pub fn parse(value: &str) -> Result<CompressionArg> {
        match value.to_ascii_lowercase().as_str() {
            "none" | "uncompressed" => Ok(CompressionArg::None),
            "snappy" => Ok(CompressionArg::Snappy),
            "zstd" => Ok(CompressionArg::Zstd),
            "gzip" => Ok(CompressionArg::Gzip),
            "brotli" => Ok(CompressionArg::Brotli),
            other => bail!("unknown compression {other:?}; expected zstd, snappy, gzip, brotli or none"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CompressionArg::None => "none",
            CompressionArg::Snappy => "snappy",
            CompressionArg::Zstd => "zstd",
            CompressionArg::Gzip => "gzip",
            CompressionArg::Brotli => "brotli",
        }
    }

    fn resolve(self, level: Option<i32>) -> Result<Compression> {
        Ok(match self {
            CompressionArg::None => Compression::UNCOMPRESSED,
            CompressionArg::Snappy => Compression::SNAPPY,
            CompressionArg::Zstd => Compression::ZSTD(match level {
                Some(level) => ZstdLevel::try_new(level)
                    .map_err(|error| anyhow::anyhow!("invalid zstd level {level}: {error}"))?,
                None => ZstdLevel::default(),
            }),
            CompressionArg::Gzip => Compression::GZIP(match level {
                Some(level) => GzipLevel::try_new(level as u32)
                    .map_err(|error| anyhow::anyhow!("invalid gzip level {level}: {error}"))?,
                None => GzipLevel::default(),
            }),
            CompressionArg::Brotli => Compression::BROTLI(match level {
                Some(level) => BrotliLevel::try_new(level as u32)
                    .map_err(|error| anyhow::anyhow!("invalid brotli level {level}: {error}"))?,
                None => BrotliLevel::default(),
            }),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictionaryArg {
    Auto,
    On,
    Off,
}

impl DictionaryArg {
    pub fn parse(value: &str) -> Result<DictionaryArg> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(DictionaryArg::Auto),
            "on" | "true" | "1" => Ok(DictionaryArg::On),
            "off" | "false" | "0" => Ok(DictionaryArg::Off),
            other => bail!("unknown dictionary setting {other:?}; expected auto, on or off"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WriterOptions {
    pub compression: CompressionArg,
    pub compression_level: Option<i32>,
    /// Hard row-count limit per row group; streaming chunks align to these.
    pub row_group_rows: usize,
    /// Byte ceiling per row group, so long-row corpora stay seekable.
    pub row_group_bytes: usize,
    pub dictionary: DictionaryArg,
    /// Text corpora gain nothing from min/max on multi-kilobyte strings and
    /// pay for it in footer size, so statistics are off unless asked for.
    pub statistics: bool,
    /// ``--align-chunk-bytes``: when set, row groups are cut only by the
    /// caller's explicit flushes, so one row group *is* one streaming chunk.
    pub aligned_chunk_bytes: Option<u64>,
}

pub const DEFAULT_ROW_GROUP_BYTES: usize = 128 * 1024 * 1024;

impl Default for WriterOptions {
    fn default() -> Self {
        WriterOptions {
            compression: CompressionArg::Zstd,
            compression_level: None,
            row_group_rows: 65_536,
            row_group_bytes: DEFAULT_ROW_GROUP_BYTES,
            dictionary: DictionaryArg::Auto,
            statistics: false,
            aligned_chunk_bytes: None,
        }
    }
}

impl WriterOptions {
    /// Human-readable compression label for the report.
    pub fn compression_label(&self) -> String {
        match self.compression_level {
            Some(level) => format!("{}({level})", self.compression.name()),
            None => self.compression.name().to_string(),
        }
    }

    fn properties(&self, schema: &SchemaRef, metadata: &[(String, String)]) -> Result<WriterProperties> {
        let aligned = self.aligned_chunk_bytes.is_some();
        let mut builder = WriterProperties::builder()
            .set_compression(self.compression.resolve(self.compression_level)?)
            .set_statistics_enabled(if self.statistics {
                EnabledStatistics::Chunk
            } else {
                EnabledStatistics::None
            })
            .set_dictionary_enabled(!matches!(self.dictionary, DictionaryArg::Off))
            .set_created_by(format!("dataset_compiler {}", env!("CARGO_PKG_VERSION")));
        if aligned {
            // One row group per aligned chunk: the caller decides where groups
            // end so a chunk maps onto exactly one group.
            builder = builder
                .set_max_row_group_row_count(None)
                .set_max_row_group_bytes(None);
        } else {
            builder = builder
                .set_max_row_group_row_count(Some(self.row_group_rows.max(1)))
                .set_max_row_group_bytes(Some(self.row_group_bytes.max(1)));
        }
        if !metadata.is_empty() {
            builder = builder.set_key_value_metadata(Some(
                metadata
                    .iter()
                    .map(|(key, value)| KeyValue::new(key.clone(), value.clone()))
                    .collect(),
            ));
        }
        if self.dictionary == DictionaryArg::Auto && schema.field_with_name("text").is_ok() {
            // A corpus text column is effectively unique per row, so a
            // dictionary only buffers and re-encodes it.  Every other column
            // (roles, token counts, flags) keeps dictionary encoding.
            builder = builder
                .set_column_dictionary_enabled(ColumnPath::new(vec!["text".to_string()]), false);
        }
        Ok(builder.build())
    }
}

#[derive(Debug, Default, Clone)]
struct HashState {
    hasher: Sha256,
    bytes: u64,
}

/// Wraps the file so the digest and byte count describe what was written.
struct HashingWriter {
    inner: BufWriter<File>,
    state: Arc<Mutex<HashState>>,
}

impl Write for HashingWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buffer)?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.hasher.update(&buffer[..written]);
        state.bytes += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Debug, Clone)]
pub struct SinkStats {
    pub rows: u64,
    pub bytes: u64,
    pub row_groups: usize,
    pub sha256: String,
}

pub struct ParquetSink {
    writer: ArrowWriter<HashingWriter>,
    state: Arc<Mutex<HashState>>,
    rows: u64,
}

impl ParquetSink {
    pub fn create(
        path: &Path,
        schema: SchemaRef,
        options: &WriterOptions,
        metadata: &[(String, String)],
    ) -> Result<ParquetSink> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("cannot create {}", parent.display()))?;
            }
        }
        let file = File::create(path)
            .with_context(|| format!("cannot create {}", path.display()))?;
        let state = Arc::new(Mutex::new(HashState::default()));
        let hashing = HashingWriter {
            inner: BufWriter::with_capacity(WRITE_BUFFER_BYTES * 8, file),
            state: Arc::clone(&state),
        };
        let writer = ArrowWriter::try_new(
            hashing,
            Arc::clone(&schema),
            Some(options.properties(&schema, metadata)?),
        )
        .with_context(|| format!("cannot start a parquet writer for {}", path.display()))?;
        Ok(ParquetSink {
            writer,
            state,
            rows: 0,
        })
    }

    /// Close the current row group.  ``--align-chunk-bytes`` calls this at each
    /// chunk boundary so a row group is exactly one streaming chunk.
    pub fn flush_row_group(&mut self) -> Result<()> {
        self.writer.flush().context("cannot close a row group")
    }

    pub fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        self.rows += batch.num_rows() as u64;
        self.writer
            .write(batch)
            .context("cannot write a record batch")
    }

    /// Rows handed to the writer so far; used for progress reporting.
    pub fn rows_written(&self) -> u64 {
        self.rows
    }

    /// Bytes already flushed to the file; buffered pages are not counted.
    pub fn bytes_written(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .bytes
    }

    pub fn finish(self) -> Result<SinkStats> {
        let metadata = self.writer.close().context("cannot finalize the parquet file")?;
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(SinkStats {
            rows: self.rows,
            bytes: state.bytes,
            row_groups: metadata.num_row_groups(),
            sha256: format!("{:x}", state.hasher.clone().finalize()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compression_levels_are_validated() {
        assert!(CompressionArg::parse("zstd").unwrap().resolve(Some(3)).is_ok());
        assert!(CompressionArg::parse("zstd").unwrap().resolve(Some(99)).is_err());
        assert!(CompressionArg::parse("lz4").is_err());
        assert_eq!(
            CompressionArg::parse("ZSTD").unwrap(),
            CompressionArg::Zstd
        );
    }

    #[test]
    fn auto_dictionary_disables_only_the_text_column() {
        let schema: SchemaRef = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("text", arrow_schema::DataType::Utf8, true),
            arrow_schema::Field::new("token_count", arrow_schema::DataType::Int64, true),
        ]));
        let text = ColumnPath::new(vec!["text".to_string()]);
        let counts = ColumnPath::new(vec!["token_count".to_string()]);

        let auto = WriterOptions::default().properties(&schema, &[]).unwrap();
        assert!(!auto.dictionary_enabled(&text));
        assert!(auto.dictionary_enabled(&counts));

        let off = WriterOptions {
            dictionary: DictionaryArg::Off,
            ..WriterOptions::default()
        }
        .properties(&schema, &[])
        .unwrap();
        assert!(!off.dictionary_enabled(&text));
        assert!(!off.dictionary_enabled(&counts));
    }
}
