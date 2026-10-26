//! Compile report sidecar: the audit trail for one compiled corpus.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Bytes hashed from each end of the source.  The full-file digest already
/// lives in the streaming plan; this only has to identify the revision.
const SAMPLE_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct InputSummary {
    pub path: String,
    pub bytes: u64,
    pub mtime_ns: u64,
    pub sampled_sha256: String,
}

impl InputSummary {
    pub fn read(path: &Path) -> Result<InputSummary> {
        let mut file = File::open(path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        let metadata = file
            .metadata()
            .with_context(|| format!("cannot stat {}", path.display()))?;
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        let mut sample = Vec::with_capacity((SAMPLE_BYTES * 2) as usize);
        let head = SAMPLE_BYTES.min(metadata.len());
        let mut head_buffer = vec![0u8; head as usize];
        file.read_exact(&mut head_buffer)
            .with_context(|| format!("cannot read {}", path.display()))?;
        sample.extend_from_slice(&head_buffer);
        if metadata.len() > head {
            use std::io::Seek;
            let tail = SAMPLE_BYTES.min(metadata.len() - head);
            file.seek(std::io::SeekFrom::End(-(tail as i64)))
                .with_context(|| format!("cannot seek {}", path.display()))?;
            let mut tail_buffer = vec![0u8; tail as usize];
            file.read_exact(&mut tail_buffer)
                .with_context(|| format!("cannot read {}", path.display()))?;
            sample.extend_from_slice(&tail_buffer);
        }
        Ok(InputSummary {
            path: normalize(path),
            bytes: metadata.len(),
            mtime_ns,
            sampled_sha256: format!("{:x}", Sha256::digest(&sample)),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputSummary {
    pub path: String,
    pub bytes: u64,
    pub rows: u64,
    pub row_groups: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColumnSummary {
    pub name: String,
    #[serde(rename = "type")]
    pub data_type: String,
    pub nullable: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub name: String,
    pub created_at: String,
    pub tool: String,
    pub format: String,
    pub input: InputSummary,
    pub output: OutputSummary,
    pub compression: String,
    pub row_group_rows: usize,
    pub batch_rows: usize,
    pub threads: usize,
    /// Rows the schema was derived from: the whole file when pre-scanning.
    pub schema_scanned_rows: u64,
    /// Set when the file was cut into one row group per streaming chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aligned_chunk_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aligned_chunks: Option<usize>,
    /// Rows kept as all-null placeholders because they could not be parsed.
    pub skipped_rows: u64,
    pub elapsed_seconds: f64,
    pub throughput_mb_per_s: f64,
    pub columns: Vec<ColumnSummary>,
    pub warnings: Vec<String>,
}

impl Report {
    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("cannot create {}", parent.display()))?;
            }
        }
        let text = serde_json::to_string_pretty(self).context("cannot render the report")?;
        std::fs::write(path, format!("{text}\n"))
            .with_context(|| format!("cannot write {}", path.display()))
    }
}

/// Repository-relative paths keep reports comparable across machines.
pub fn normalize(path: &Path) -> String {
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let text = resolved.to_string_lossy().to_string();
    text.strip_prefix(r"\\?\").unwrap_or(&text).to_string()
}

pub fn iso8601_now() -> String {
    iso8601(SystemTime::now())
}

pub fn iso8601(time: SystemTime) -> String {
    let duration = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let seconds = duration.as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let in_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        in_day / 3600,
        (in_day % 3600) / 60,
        in_day % 60,
        duration.subsec_millis()
    )
}

/// Howard Hinnant's ``civil_from_days``: days since 1970-01-01 to a date.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn epoch_and_known_dates_convert() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_720), (2026, 9, 24));
        assert_eq!(
            iso8601(UNIX_EPOCH + Duration::from_millis(1_759_000_000_123)),
            "2025-09-27T19:06:40.123Z"
        );
    }
}
