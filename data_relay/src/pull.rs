//! Pull-channel (高速通道) artifacts: the relay serves task output over HTTP
//! with Range support; the no-GPU instance pulls the data itself with
//! aria2c/curl. SSH/SFTP carries only control-plane traffic (quota pre-check,
//! manifest upload, command dispatch, progress), never bulk data.

#[cfg(test)]
#[cfg(test)]
#[cfg(test)]
use crate::model;
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use std::path::Path;

/// Generate a fresh per-task pull token (uuid v4 without dashes).
pub fn new_pull_token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Encode a relative path for use inside a URL, preserving `/`.
fn encode_rel(rel: &str) -> String {
    const SET: &AsciiSet = &CONTROLS
        .add(b' ')
        .add(b'"')
        .add(b'#')
        .add(b'%')
        .add(b'&')
        .add(b'\'')
        .add(b'+')
        .add(b',')
        .add(b':')
        .add(b';')
        .add(b'<')
        .add(b'=')
        .add(b'>')
        .add(b'?')
        .add(b'@')
        .add(b'[')
        .add(b']')
        .add(b'\\')
        .add(b'^')
        .add(b'`')
        .add(b'{')
        .add(b'|')
        .add(b'}');
    rel.split('/')
        .map(|seg| utf8_percent_encode(seg, SET).to_string())
        .collect::<Vec<_>>()
        .join("/")
}

/// One file to pull: path relative to the task output dir, plus its size.
#[derive(Debug, Clone, PartialEq)]
pub struct PullFile {
    pub rel: String,
    pub size: u64,
}

/// Walk the local output dir and collect every file for the pull manifest.
pub fn scan_local(local_root: &Path) -> anyhow::Result<Vec<PullFile>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(local_root) {
        let entry = entry.map_err(|e| anyhow::anyhow!("遍历 {} 失败: {e}", local_root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(local_root)
            .map_err(|e| anyhow::anyhow!("strip_prefix 失败: {e}"))?
            .to_string_lossy()
            .replace('\\', "/");
        if rel.is_empty() || rel.starts_with("_relay") || rel.ends_with(".rpart") {
            continue; // control-plane files never travel through the pull channel
        }
        out.push(PullFile {
            rel,
            size: entry.metadata()?.len(),
        });
    }
    Ok(out)
}

/// TSV manifest: `rel<TAB>size` per line — used for progress accounting and
/// size verification on both sides.
pub fn build_manifest_tsv(files: &[PullFile]) -> String {
    let mut s = String::from("# relay manifest v1: rel\\tsize\n");
    for f in files {
        s.push_str(&format!("{}\t{}\n", f.rel, f.size));
    }
    s
}

/// aria2 input file: one URI block per file (per-download `dir=`/`out=`).
/// `remote_task_root` is where files land on the instance.
pub fn build_aria2_input(
    base_url: &str,
    task_id: &str,
    token: &str,
    remote_task_root: &str,
    files: &[PullFile],
) -> String {
    let base = base_url.trim_end_matches('/');
    let root = remote_task_root.trim_end_matches('/');
    let mut s = String::new();
    for f in files {
        let url = format!("{base}/pull/{task_id}/{}?t={token}", encode_rel(&f.rel));
        s.push_str(&url);
        s.push('\n');
        let (dir, file) = match f.rel.rsplit_once('/') {
            Some((parent, name)) => (format!("{root}/{parent}"), name.to_string()),
            None => (root.to_string(), f.rel.clone()),
        };
        s.push_str(&format!("  dir={dir}\n  out={file}\n"));
    }
    s
}

/// The bash script written to the instance and launched with nohup. Prefers
/// aria2c (multi-connection), falls back to curl -C - (resumable).
pub fn build_pull_script(
    base_url: &str,
    task_id: &str,
    token: &str,
    remote_task_root: &str,
) -> String {
    let base = base_url.trim_end_matches('/');
    format!(
        r#"#!/bin/bash
# data_relay pull script for task {task_id}
set -u
ROOT="{root}"
LOG="$ROOT/_relay_pull.log"
BASE="{base}/pull/{task_id}"
T="{token}"

mkdir -p "$ROOT"
cd "$ROOT"
: > "$LOG"
echo "[relay-pull] start $(date -u +%FT%TZ)" >> "$LOG"

rc=0
if command -v aria2c >/dev/null 2>&1; then
  echo "[relay-pull] engine=aria2c" >> "$LOG"
  aria2c -i "$ROOT/_relay_manifest.aria2" \
    --continue=true --allow-overwrite=true --auto-file-renaming=false \
    --file-allocation=none --max-connection-per-server=16 --split=16 \
    --min-split-size=1M --max-concurrent-downloads=8 \
    --summary-interval=5 --console-log-level=warn >> "$LOG" 2>&1
  rc=$?
else
  echo "[relay-pull] engine=curl" >> "$LOG"
  while IFS=$'\t' read -r rel size; do
    case "$rel" in ''|\#*) continue ;; esac
    mkdir -p "$ROOT/$(dirname "$rel")"
    echo "[relay-pull] pulling $rel ($size bytes)" >> "$LOG"
    curl -sfL -C - -o "$ROOT/$rel" "$BASE/$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$rel")?t=$T" || {{ rc=1; echo "[relay-pull] FAILED $rel" >> "$LOG"; break; }}
    echo "[relay-pull] done $rel" >> "$LOG"
  done < "$ROOT/_relay_manifest.tsv"
fi

# size verification against the manifest
bad=0
while IFS=$'\t' read -r rel size; do
  case "$rel" in ''|\#*) continue ;; esac
  actual=$(stat -c %s "$ROOT/$rel" 2>/dev/null || echo -1)
  if [ "$actual" != "$size" ]; then
    echo "[relay-pull] size-mismatch $rel expected=$size actual=$actual" >> "$LOG"
    bad=1
  fi
done < "$ROOT/_relay_manifest.tsv"
[ "$bad" = 1 ] && rc=1

echo "RELAY-PULL-EXIT=$rc" >> "$LOG"
echo "[relay-pull] finish $(date -u +%FT%TZ) rc=$rc" >> "$LOG"
"#,
        task_id = task_id,
        root = remote_task_root.trim_end_matches('/'),
        base = base,
        token = token,
    )
}

/// Total bytes across a manifest.
pub fn manifest_total(files: &[PullFile]) -> u64 {
    files.iter().map(|f| f.size).sum()
}

/// Parse a manifest TSV back into entries (used by tests and the size check).
pub fn parse_manifest_tsv(text: &str) -> Vec<PullFile> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (rel, size) = l.split_once('\t')?;
            size.parse().ok().map(|s| PullFile {
                rel: rel.to_string(),
                size: s,
            })
        })
        .collect()
}

/// Local control-plane file names (never uploaded as data).
pub const MANIFEST_TSV: &str = "_relay_manifest.tsv";
pub const MANIFEST_ARIA2: &str = "_relay_manifest.aria2";
pub const PULL_SCRIPT: &str = "_relay_pull.sh";
pub const PULL_LOG: &str = "_relay_pull.log";
pub const PULL_DONE_MARKER: &str = "RELAY-PULL-EXIT=";

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> Vec<PullFile> {
        vec![
            PullFile {
                rel: "big.bin".into(),
                size: 123456,
            },
            PullFile {
                rel: "sub/dir/data.jsonl".into(),
                size: 789,
            },
        ]
    }

    #[test]
    fn manifest_roundtrip() {
        let tsv = build_manifest_tsv(&files());
        assert_eq!(parse_manifest_tsv(&tsv), files());
        assert_eq!(manifest_total(&files()), 124245);
    }

    #[test]
    fn scan_local_skips_control_files() {
        let tmp = std::env::temp_dir().join(format!("relay_pull_{}", uuid::Uuid::new_v4()));
        let out = tmp.join("out");
        std::fs::create_dir_all(out.join("sub")).unwrap();
        std::fs::write(out.join("a.bin"), [0u8; 10]).unwrap();
        std::fs::write(out.join("sub").join("b.txt"), [0u8; 5]).unwrap();
        std::fs::write(out.join("_relay_pull.log"), b"log").unwrap();
        std::fs::write(out.join("c.rpart"), b"partial").unwrap();
        let files = scan_local(&out).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| !f.rel.starts_with("_relay")));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn aria2_input_has_dir_and_out() {
        let input = build_aria2_input(
            "http://1.2.3.4:7800/",
            "t_x",
            "tok",
            "/root/autodl-tmp/relay_uploads/t_x",
            &files(),
        );
        assert!(input.contains("http://1.2.3.4:7800/pull/t_x/big.bin?t=tok\n  dir=/root/autodl-tmp/relay_uploads/t_x\n  out=big.bin\n"));
        assert!(input.contains("/pull/t_x/sub/dir/data.jsonl?t=tok\n  dir=/root/autodl-tmp/relay_uploads/t_x/sub/dir\n  out=data.jsonl\n"));
    }

    #[test]
    fn pull_script_contains_markers() {
        let s = build_pull_script("http://r:7800", "t_1", "tk", "/root/x/t_1");
        assert!(s.contains("RELAY-PULL-EXIT="));
        assert!(s.contains("--max-connection-per-server=16"));
        assert!(s.contains("size-mismatch"));
    }

    #[test]
    fn token_is_url_safe() {
        let t = new_pull_token();
        assert_eq!(t.len(), 32);
        assert!(t.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn model_join_remote_still_guards() {
        assert!(model::join_remote("/root/x", "../y").is_none());
    }
}
