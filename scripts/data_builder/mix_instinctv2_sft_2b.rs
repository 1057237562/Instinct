//! Deduplicate audited SFT candidates, then shuffle and write aligned dataset shards.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread;

const SHARDS: usize = 32;
const BUFFER: usize = 1024 * 1024;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn open_output(path: &Path) -> io::Result<BufWriter<File>> {
    Ok(BufWriter::with_capacity(
        BUFFER, OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn number<T: std::str::FromStr>(field: &[u8]) -> io::Result<T> {
    std::str::from_utf8(field).ok().and_then(|text| text.parse().ok())
        .ok_or_else(|| invalid("invalid numeric candidate field"))
}

fn hash(field: &[u8]) -> io::Result<u128> {
    std::str::from_utf8(field).ok().and_then(|text| u128::from_str_radix(text, 16).ok())
        .ok_or_else(|| invalid("invalid candidate fingerprint"))
}

fn contest_group(provenance: &[u8]) -> io::Result<u128> {
    const KEY: &[u8] = b"\"canonical_problem_sha256\":\"";
    const CPP: &[u8] = b"\"language\":\"cpp\"";
    let start = provenance.windows(KEY.len()).position(|part| part == KEY)
        .ok_or_else(|| invalid("missing canonical competitive problem"))? + KEY.len();
    let digest = provenance.get(start..start + 32).ok_or_else(|| invalid("short competitive problem fingerprint"))?;
    let language = if provenance.windows(CPP.len()).any(|part| part == CPP) { 1 } else { 0 };
    Ok(hash(digest)? ^ language)
}

struct Record<'a> {
    prompt: u128,
    group: u128,
    answer: Option<u128>,
    tokens: u64,
    supervised: u64,
    category: &'a str,
    flags: u8,
    data: &'a [u8],
    provenance: &'a [u8],
}

impl<'a> Record<'a> {
    fn parse(line: &'a [u8]) -> io::Result<Self> {
        let line = line.strip_suffix(b"\n").ok_or_else(|| invalid("unterminated candidate"))?;
        let fields: Vec<_> = line.splitn(9, |&byte| byte == b'\t').collect();
        if fields.len() != 9 || !fields[7].ends_with(b"}") || !fields[8].ends_with(b"}") {
            return Err(invalid("invalid candidate record"));
        }
        let record = Self {
            prompt: hash(fields[0])?,
            group: hash(fields[1])?,
            answer: if fields[2].is_empty() { None } else { Some(hash(fields[2])?) },
            tokens: number(fields[3])?,
            supervised: number(fields[4])?,
            category: std::str::from_utf8(fields[5]).map_err(|_| invalid("invalid category"))?,
            flags: number(fields[6])?,
            data: fields[7],
            provenance: fields[8],
        };
        if record.tokens < 12 || record.tokens > 16384 || record.supervised == 0
            || record.supervised > record.tokens || record.flags > 3
        {
            return Err(invalid("invalid candidate counts"));
        }
        Ok(record)
    }
}

#[derive(Default)]
struct Totals {
    rows: u64,
    bytes: u64,
    tokens: u64,
    supervised: u64,
    thinking: u64,
    tools: u64,
    longest: u64,
}

impl Totals {
    fn add(&mut self, record: &Record) {
        self.rows += 1;
        self.bytes += record.data.len() as u64 + 1;
        self.tokens += record.tokens;
        self.supervised += record.supervised;
        self.thinking += (record.flags & 1) as u64;
        self.tools += ((record.flags >> 1) & 1) as u64;
        self.longest = self.longest.max(record.tokens);
    }
}

#[derive(Default)]
struct Mix {
    prompts: HashMap<u128, u8>,
    groups: HashMap<u128, u8>,
    answers: HashSet<u128>,
    validation_prompts: HashSet<u128>,
    validation_groups: HashSet<u128>,
    categories: BTreeMap<String, Totals>,
    totals: Totals,
    duplicates: u64,
    oversized_for_budget: u64,
    byte_budget: bool,
}

fn scan(path: &Path, mut handle: impl FnMut(&[u8], &Record) -> io::Result<bool>) -> io::Result<()> {
    let mut input = BufReader::with_capacity(BUFFER, File::open(path)?);
    let mut line = Vec::new();
    while input.read_until(b'\n', &mut line)? != 0 {
        let record = Record::parse(&line)?;
        if !handle(&line, &record)? {
            break;
        }
        line.clear();
    }
    Ok(())
}

fn mixed(mut key: u64) -> u64 {
    key = (key ^ (key >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    key = (key ^ (key >> 27)).wrapping_mul(0x94d049bb133111eb);
    key ^ (key >> 31)
}

impl Mix {
    fn size(&self) -> u64 {
        if self.byte_budget { self.totals.bytes } else { self.totals.tokens }
    }

    fn select(&mut self, line: &[u8], row: &Record, pinned: bool, limit: u64, group_limit: u8,
              writers: &mut [BufWriter<File>]) -> io::Result<()> {
        let cross_split = self.validation_prompts.contains(&row.prompt)
            || (!pinned && self.validation_groups.contains(&row.group));
        let group = if self.byte_budget && row.category.starts_with("contest_") {
            contest_group(row.provenance)?
        } else {
            row.group
        };
        let group_count = self.groups.get(&group).copied().unwrap_or_default();
        let prompt_count = self.prompts.get(&row.prompt).copied().unwrap_or_default();
        let duplicate_prompt = prompt_count >= group_limit;
        if pinned && (cross_split || duplicate_prompt) {
            return Err(invalid("verified baseline contains a validation overlap or duplicate prompt"));
        }
        if cross_split || duplicate_prompt
            || (!pinned && (group_count >= group_limit
                            || row.answer.is_some_and(|key| self.answers.contains(&key))))
        {
            self.duplicates += 1;
            return Ok(());
        }
        let size = if self.byte_budget { row.data.len() as u64 + 1 } else { row.tokens };
        if size > limit {
            self.oversized_for_budget += 1;
            return Ok(());
        }
        self.prompts.insert(row.prompt, prompt_count.saturating_add(1));
        self.groups.insert(group, group_count.saturating_add(1));
        if let Some(key) = row.answer {
            self.answers.insert(key);
        }
        let lane = mixed(row.prompt as u64 ^ 20261109) as usize % SHARDS;
        writers[lane].write_all(line)?;
        self.totals.add(row);
        if let Some(totals) = self.categories.get_mut(row.category) {
            totals.add(row);
        } else {
            let mut totals = Totals::default();
            totals.add(row);
            self.categories.insert(row.category.into(), totals);
        }
        Ok(())
    }
}

struct PlanEntry {
    group: String,
    quota: u64,
    path: PathBuf,
    min_tokens: u64,
    max_tokens: u64,
    group_limit: u8,
}

fn plan(path: &Path) -> io::Result<Vec<PlanEntry>> {
    fs::read_to_string(path)?.lines().map(|line| {
        let mut fields = line.split('\t');
        let group = fields.next().ok_or_else(|| invalid("missing plan group"))?;
        let quota = fields.next().ok_or_else(|| invalid("missing plan quota"))?;
        let path = fields.next().ok_or_else(|| invalid("missing plan path"))?;
        let min_tokens = fields.next().map(|field| number(field.as_bytes())).transpose()?.unwrap_or(0);
        let max_tokens = fields.next().map(|field| number(field.as_bytes())).transpose()?.unwrap_or(16384);
        let group_limit = fields.next().map(|field| number(field.as_bytes())).transpose()?.unwrap_or(1);
        if min_tokens > max_tokens || !matches!(group_limit, 1 | 2) {
            return Err(invalid("invalid token band or per-question limit"));
        }
        Ok(PlanEntry { group: group.into(), quota: number(quota.as_bytes())?, path: path.into(), min_tokens, max_tokens, group_limit })
    }).collect()
}

fn select_plan(mix: &mut Mix, entries: &[PlanEntry], target: u64, cap_groups: bool,
               writers: &mut [BufWriter<File>]) -> io::Result<()> {
    let mut used = BTreeMap::<&str, u64>::new();
    for entry in entries {
        let group = used.entry(&entry.group).or_default();
        let max = if cap_groups { entry.quota } else { target };
        if max.saturating_sub(*group) < 12 || target.saturating_sub(mix.size()) < 12 {
            continue;
        }
        scan(&entry.path, |line, record| {
            let limit = max.saturating_sub(*group).min(target.saturating_sub(mix.size()));
            if limit < 12 {
                return Ok(false);
            }
            if !(entry.min_tokens..=entry.max_tokens).contains(&record.tokens) {
                return Ok(true);
            }
            let before = mix.size();
            mix.select(line, record, false, limit, entry.group_limit, writers)?;
            *group += mix.size() - before;
            Ok(true)
        })?;
    }
    for (group, size) in used {
        eprintln!("[mix {}] {group}: {size} new {}", if cap_groups { "quota" } else { "fill" },
                  if mix.byte_budget { "bytes" } else { "tokens" });
    }
    Ok(())
}

fn shuffle_shard(lane: usize, scratch: &Path, output: &Path) -> io::Result<(usize, u64)> {
    let input = BufReader::with_capacity(BUFFER, File::open(scratch.join(format!("{lane:02}.tsv")))?);
    let mut rows = input.split(b'\n').collect::<io::Result<Vec<_>>>()?;
    let mut seed = 20261109 + lane as u64;
    for i in (1..rows.len()).rev() {
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        rows.swap(i, (mixed(seed) % (i as u64 + 1)) as usize);
    }
    let mut data = open_output(&output.join("train_jsonl").join(format!("part-{lane:02}.jsonl")))?;
    let mut provenance = open_output(&output.join("provenance").join(format!("part-{lane:02}.jsonl")))?;
    let mut tokens = 0;
    for (index, line) in rows.iter_mut().enumerate() {
        line.push(b'\n');
        let record = Record::parse(line)?;
        data.write_all(record.data)?;
        data.write_all(b"\n")?;
        provenance.write_all(&record.provenance[..record.provenance.len() - 1])?;
        writeln!(provenance, ",\"release_row\":{}}}", index + 1)?;
        tokens += record.tokens;
    }
    data.flush()?;
    provenance.flush()?;
    eprintln!("[shuffle] shard {lane:02}: {} rows, {tokens} tokens", rows.len());
    Ok((rows.len(), tokens))
}

fn main() -> io::Result<()> {
    let mut args: Vec<_> = env::args_os().skip(1).collect();
    let byte_budget = args.first().is_some_and(|arg| arg == "--bytes");
    if byte_budget {
        args.remove(0);
    }
    let [training, validation, plan_path, target, scratch, output] = &args[..] else {
        return Err(invalid("usage: mix_sft [--bytes] BASE_TRAIN BASE_VALIDATION PLAN TARGET SCRATCH OUTPUT"));
    };
    let target: u64 = number(target.to_str().ok_or_else(|| invalid("invalid target"))?.as_bytes())?;
    let scratch = Path::new(scratch);
    let output = Path::new(output);
    fs::create_dir(scratch)?;
    let mut writers = (0..SHARDS).map(|lane| open_output(&scratch.join(format!("{lane:02}.tsv"))))
        .collect::<io::Result<Vec<_>>>()?;
    let mut mix = Mix { byte_budget, ..Mix::default() };
    scan(Path::new(validation), |_, row| {
        mix.validation_prompts.insert(row.prompt);
        mix.validation_groups.insert(row.group);
        Ok(true)
    })?;
    scan(Path::new(training), |line, row| {
        mix.select(line, row, true, target.saturating_sub(mix.size()), 1, &mut writers)?;
        Ok(true)
    })?;
    let entries = plan(Path::new(plan_path))?;
    select_plan(&mut mix, &entries, target, true, &mut writers)?;
    if !byte_budget && target.saturating_sub(mix.size()) >= 12 {
        select_plan(&mut mix, &entries, target, false, &mut writers)?;
    }
    for writer in &mut writers {
        writer.flush()?;
    }
    drop(writers);
    println!("summary\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
             mix.totals.rows, mix.totals.tokens, mix.totals.supervised, mix.totals.thinking,
             mix.totals.tools, mix.totals.longest, mix.duplicates, mix.oversized_for_budget);
    for (category, totals) in &mix.categories {
        println!("category\t{category}\t{}\t{}\t{}\t{}\t{}\t{}",
                 totals.rows, totals.tokens, totals.supervised, totals.thinking, totals.tools, totals.longest);
        if byte_budget {
            println!("bytes\t{category}\t{}", totals.bytes);
        }
    }
    if byte_budget {
        println!("bytes\ttotal\t{}", mix.totals.bytes);
    } else if target.saturating_sub(mix.size()) > 1000 {
        return Err(invalid("eligible unique candidate tokens did not reach the requested budget"));
    }
    fs::create_dir(output)?;
    fs::create_dir(output.join("train_jsonl"))?;
    fs::create_dir(output.join("provenance"))?;
    let (rows, tokens) = thread::scope(|scope| -> io::Result<_> {
        let workers: Vec<_> = (0..8).map(|worker| scope.spawn(move || -> io::Result<(u64, u64)> {
            let (mut rows, mut tokens) = (0, 0);
            for lane in (worker..SHARDS).step_by(8) {
                let (count, sum) = shuffle_shard(lane, scratch, output)?;
                rows += count as u64;
                tokens += sum;
            }
            Ok((rows, tokens))
        })).collect();
        let (mut rows, mut tokens) = (0, 0);
        for worker in workers {
            let (count, sum) = worker.join().map_err(|_| invalid("shuffle worker panicked"))??;
            rows += count;
            tokens += sum;
        }
        Ok((rows, tokens))
    })?;
    if (rows, tokens) != (mix.totals.rows, mix.totals.tokens) {
        return Err(invalid("shuffle row or token conservation failed"));
    }
    Ok(())
}
