//! Compare a compiled parquet file against its JSONL source.
//!
//! The comparison is schema driven: source rows are normalised through the same
//! column model the compiler used (read back from the parquet footer), so the
//! check measures the conversion itself rather than two JSON serializations.
//! Row counts are compared exactly, because a trainer that loses rows silently
//! is the failure this tool exists to prevent.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::build::{cell_to_value, convert_row};
use crate::read::{canonical, plan_from_arrow, read_values, require_columns, summarize};
use crate::schema::Preset;

#[derive(Debug, Clone)]
pub struct VerifyOptions {
    pub format: Preset,
    pub auto_detect: bool,
    pub text_key: String,
    pub rows: u64,
    pub offset: u64,
    pub allow_row_mismatch: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        VerifyOptions {
            format: Preset::Generic,
            auto_detect: true,
            text_key: "text".to_string(),
            rows: 1000,
            offset: 0,
            allow_row_mismatch: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerifyOutcome {
    pub source_rows: u64,
    pub compiled_rows: i64,
    pub compared: u64,
    pub mismatches: Vec<String>,
    pub row_count_ok: bool,
}

impl VerifyOutcome {
    pub fn ok(&self) -> bool {
        self.mismatches.is_empty() && self.row_count_ok
    }

    pub fn summary(&self) -> String {
        format!(
            "source_rows={} compiled_rows={} compared={} value_mismatches={} row_count={}",
            self.source_rows,
            self.compiled_rows,
            self.compared,
            self.mismatches.len(),
            if self.row_count_ok { "match" } else { "MISMATCH" }
        )
    }
}

pub fn verify(source: &Path, compiled: &Path, options: &VerifyOptions) -> Result<VerifyOutcome> {
    if !source.is_file() {
        bail!("source is not a file: {}", source.display());
    }
    let summary = summarize(compiled)?;
    let preset = if options.auto_detect {
        detect_source_preset(source, &options.text_key)?
    } else {
        options.format
    };
    let (schema, compiled_rows) = read_values(compiled, options.offset, Some(options.rows))?;
    require_columns(&schema, preset, &options.text_key)?;
    let plan = plan_from_arrow(&schema, preset)?;
    let expected = read_source_sample(source, &plan, options)?;

    let mut mismatches = Vec::new();
    let compared = expected.rows.len().min(compiled_rows.len()) as u64;
    for index in 0..compared as usize {
        let left = canonical(&expected.rows[index]);
        let right = canonical(&compiled_rows[index]);
        if left != right {
            let position = options.offset + index as u64 + 1;
            let detail = first_difference(&left, &right)
                .unwrap_or_else(|| "values differ".to_string());
            if mismatches.len() < 10 {
                mismatches.push(format!("row {position}: {detail}"));
            }
        }
    }
    if expected.rows.len() as i64 != compiled_rows.len() as i64 && mismatches.len() < 10 {
        mismatches.push(format!(
            "sampled {} source rows but read {} compiled rows",
            expected.rows.len(),
            compiled_rows.len()
        ));
    }
    let row_count_ok = expected.source_rows == summary.rows.max(0) as u64
        || options.allow_row_mismatch;
    if !row_count_ok {
        mismatches.push(format!(
            "source has {} rows but the parquet file has {}",
            expected.source_rows, summary.rows
        ));
    }
    Ok(VerifyOutcome {
        source_rows: expected.source_rows,
        compiled_rows: summary.rows,
        compared,
        mismatches,
        row_count_ok,
    })
}

struct SourceSample {
    source_rows: u64,
    rows: Vec<Value>,
}

/// One sequential pass: count every non-blank row, and convert the window
/// ``[offset, offset + rows)`` through the compiled schema.
fn read_source_sample(
    source: &Path,
    plan: &crate::schema::SchemaPlan,
    options: &VerifyOptions,
) -> Result<SourceSample> {
    let file = File::open(source).with_context(|| format!("cannot open {}", source.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut buffer: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut source_rows = 0u64;
    let mut rows = Vec::new();
    let window_end = options.offset.saturating_add(options.rows);
    loop {
        buffer.clear();
        if reader
            .read_until(b'\n', &mut buffer)
            .with_context(|| format!("reading {}", source.display()))?
            == 0
        {
            break;
        }
        while matches!(buffer.last(), Some(b'\n') | Some(b'\r')) {
            buffer.pop();
        }
        if buffer.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        let position = source_rows;
        source_rows += 1;
        if position < options.offset || position >= window_end {
            continue;
        }
        let value: Value = serde_json::from_slice(&buffer)
            .with_context(|| format!("row {}: invalid JSON", position + 1))?;
        rows.push(cell_to_value(&convert_row(&value, plan)?, plan));
    }
    Ok(SourceSample { source_rows, rows })
}

fn detect_source_preset(source: &Path, text_key: &str) -> Result<Preset> {
    Ok(crate::schema::detect_preset(
        &crate::reader::first_row(source)?,
        text_key,
    ))
}

/// Describe where two canonical values first diverge.
fn first_difference(expected: &Value, actual: &Value) -> Option<String> {
    match (expected, actual) {
        (Value::Object(left), Value::Object(right)) => {
            for key in left.keys() {
                if !right.contains_key(key) {
                    return Some(format!("missing key {key:?}"));
                }
            }
            for key in right.keys() {
                if !left.contains_key(key) {
                    return Some(format!("unexpected key {key:?}"));
                }
            }
            for (key, value) in left {
                if let Some(detail) = first_difference(value, &right[key]) {
                    return Some(format!("{key}.{detail}"));
                }
            }
            None
        }
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                return Some(format!("array length {} != {}", left.len(), right.len()));
            }
            for (index, value) in left.iter().enumerate() {
                if let Some(detail) = first_difference(value, &right[index]) {
                    return Some(format!("[{index}].{detail}"));
                }
            }
            None
        }
        (left, right) => {
            let left = render(left);
            let right = render(right);
            if left == right {
                None
            } else {
                Some(format!("expected {left}, found {right}"))
            }
        }
    }
}

fn render(value: &Value) -> String {
    let text = value.to_string();
    if text.len() <= 120 {
        return text;
    }
    // Truncate on a character boundary so multi-byte corpus text stays valid.
    let mut end = 120;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… ({} bytes)", &text[..end], text.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile_file, CompileOptions};
    use std::io::Write;

    /// Compile ``lines`` and keep both files alive in the returned directory.
    fn compile(lines: &[&str], options: CompileOptions) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("source.jsonl");
        let mut file = File::create(&input).unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
        file.flush().unwrap();
        drop(file);
        let output = directory.path().join("out.parquet");
        compile_file(&input, &output, &options).unwrap();
        (directory, output)
    }

    fn quiet() -> CompileOptions {
        CompileOptions {
            quiet: true,
            progress: false,
            threads: 2,
            ..CompileOptions::default()
        }
    }

    #[test]
    fn compiled_chat_output_matches_its_source() {
        let (directory, output) = compile(
            &[
                "{\"conversations\": [{\"role\": \"user\", \"content\": \"hi\", \"tools\": {\"a\": 1}}]}",
                "{\"conversations\": []}",
                "{\"conversations\": [{\"role\": \"assistant\", \"content\": \"ok\"}]}",
            ],
            CompileOptions {
                format: Preset::Sft,
                auto_detect: false,
                ..quiet()
            },
        );
        let outcome = verify(
            &directory.path().join("source.jsonl"),
            &output,
            &VerifyOptions {
                format: Preset::Sft,
                auto_detect: false,
                rows: 100,
                ..VerifyOptions::default()
            },
        )
        .unwrap();
        assert!(outcome.ok(), "{:?}", outcome.mismatches);
        assert_eq!(outcome.compared, 3);
        assert_eq!(outcome.source_rows, 3);
    }

    #[test]
    fn tail_window_is_compared() {
        let lines: Vec<String> = (0..50)
            .map(|index| format!("{{\"text\": \"row {index}\"}}"))
            .collect();
        let borrowed: Vec<&str> = lines.iter().map(String::as_str).collect();
        let (directory, output) = compile(&borrowed, quiet());
        let outcome = verify(
            &directory.path().join("source.jsonl"),
            &output,
            &VerifyOptions {
                offset: 40,
                rows: 5,
                ..VerifyOptions::default()
            },
        )
        .unwrap();
        assert!(outcome.ok(), "{:?}", outcome.mismatches);
        assert_eq!(outcome.source_rows, 50);
        assert_eq!(outcome.compared, 5);
    }

    #[test]
    fn a_missing_row_count_is_reported() {
        let (directory, output) = compile(
            &["{\"text\": \"a\"}", "{\"text\": \"b\"}", "{\"text\": \"c\"}"],
            CompileOptions {
                limit: Some(2),
                ..quiet()
            },
        );
        let outcome = verify(
            &directory.path().join("source.jsonl"),
            &output,
            &VerifyOptions::default(),
        )
        .unwrap();
        assert!(!outcome.ok());
        assert!(outcome
            .mismatches
            .iter()
            .any(|message| message.contains("source has 3 rows")));
    }

    #[test]
    fn first_difference_uses_paths() {
        let expected = serde_json::json!({"conversations": [{"role": "user"}]});
        let actual = serde_json::json!({"conversations": [{"role": "assistant"}]});
        let detail = first_difference(&expected, &actual).unwrap();
        assert!(detail.starts_with("conversations.[0].role"), "{detail}");
    }

    #[test]
    fn long_values_are_truncated_safely() {
        let value = Value::String("漢".repeat(200));
        let rendered = render(&value);
        assert!(rendered.ends_with("bytes)"), "{rendered}");
    }
}
