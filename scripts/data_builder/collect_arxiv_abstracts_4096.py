"""Collect 160M openly licensed arXiv abstract tokens with a 4096-token cap."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import orjson
from transformers import AutoTokenizer

if __package__ in (None, ""):
    __package__ = "scripts"
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.data_builder.collect_arxiv_pretrain import CountingReader, USER_AGENT, shard_url


REPO = "common-pile/arxiv_abstracts"
REVISION = "828e35d1000f94579da8850f5f640c138279bdb5"
TARGET_TOKENS = 160_000_000
MAX_TOKENS = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset/pretrain_arxiv_abstracts_160m_tokens_4096.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("dataset/pretrain_arxiv_abstracts_160m_tokens_4096.report.json"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); output=args.output.resolve(); report_path=args.report.resolve()
    temporary=output.with_name(output.name+".tmp")
    for path in (output,report_path,temporary):
        if path.exists(): raise FileExistsError(path)
    output.parent.mkdir(parents=True,exist_ok=True)
    backend=AutoTokenizer.from_pretrained(args.tokenizer_path.resolve(),trust_remote_code=True).backend_tokenizer
    digest=hashlib.sha256(); seen=set(); rows=tokens=output_bytes=0; shard_reports=[]; rejected=0
    try:
        with temporary.open("xb",buffering=8*1024*1024) as destination:
            for index in range(4):
                target=TARGET_TOKENS*(index+1)//4; path=f"{index:05d}_arxiv-abstracts.jsonl.gz"; scanned=0
                request=Request(shard_url(REPO,REVISION,path),headers={"User-Agent":USER_AGENT})
                with urlopen(request,timeout=args.timeout) as response:
                    counted=CountingReader(response)
                    with gzip.GzipFile(fileobj=counted,mode="rb") as archive:
                        for raw_line in archive:
                            scanned+=1; row=orjson.loads(raw_line); text=row.get("text"); source_id=str(row.get("id") or "")
                            if not isinstance(text,str) or not text or not source_id or source_id in seen: continue
                            token_count=len(backend.encode(text,add_special_tokens=False).ids)+2
                            if token_count>MAX_TOKENS: rejected+=1; continue
                            remaining=target-tokens
                            if token_count>remaining:
                                if remaining<MAX_TOKENS: break
                                continue
                            metadata=row.get("metadata") or {}
                            record={"text":text,"source":"common-pile/arxiv_abstracts","source_id":source_id,"license":str(metadata.get("license") or ""),"url":str(metadata.get("url") or ""),"token_count":token_count}
                            line=orjson.dumps(record,option=orjson.OPT_APPEND_NEWLINE)
                            destination.write(line);digest.update(line);seen.add(source_id);rows+=1;tokens+=token_count;output_bytes+=len(line)
                shard_reports.append({"path":path,"rows_scanned":scanned,"compressed_bytes_read":counted.bytes_read})
                print(f"[{index+1}/4] tokens={tokens:,}/{TARGET_TOKENS:,}, rows={rows:,}",flush=True)
            destination.flush();os.fsync(destination.fileno())
        os.replace(temporary,output)
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise
    report={"name":"pretrain_arxiv_abstracts_4096","created_at":datetime.now(timezone.utc).isoformat(),"source":{"repository":REPO,"revision":REVISION,"url":f"https://huggingface.co/datasets/{REPO}"},"tokenizer":str(args.tokenizer_path.resolve()),"max_tokens_including_bos_eos":MAX_TOKENS,"whole_documents_only":True,"text_split":False,"text_truncated":False,"output":str(output),"output_rows":rows,"output_tokens":tokens,"output_bytes":output_bytes,"output_sha256":digest.hexdigest(),"rejected_over_4096":rejected,"shards":shard_reports}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__": main()
