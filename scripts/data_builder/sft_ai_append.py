"""Append rewritten SFT rows to an agent output JSONL file.

Used by rewriting agents so they never hand-serialize JSON containing code blocks.

Row mode — append one pair from text files:
  python sft_ai_append.py --out <out.jsonl> --id <int> \
      --user-file <user.txt> --assistant-file <assistant.txt>

Scan mode — append every complete pair found in a group directory. Pair files are
named <id>.user.txt / <id>.assistant.txt; a <id>.skip.txt file records a skip with
its content as the reason. Appended pairs are renamed to *.done so re-runs are
idempotent:
  python sft_ai_append.py --out <out.jsonl> --scan <groupdir>

Skip mode (single row, no files):
  python sft_ai_append.py --out <out.jsonl> --id <int> --skip "<reason>"
"""
import argparse
import json
import sys
from pathlib import Path


def read_pair(directory, path):
    return path.read_text(encoding="utf-8").strip()


def scan(directory, out_path):
    directory = Path(directory)
    appended, skipped = [], []
    stems = sorted({p.name[: -len(".user.txt")] for p in directory.glob("*.user.txt")}
                   | {p.name[: -len(".skip.txt")] for p in directory.glob("*.skip.txt")})
    for article_id in stems:
        user_path = directory / f"{article_id}.user.txt"
        assistant_path = directory / f"{article_id}.assistant.txt"
        skip_path = directory / f"{article_id}.skip.txt"
        if (directory / f"{article_id}.user.done").exists() \
                or (directory / f"{article_id}.skip.done").exists():
            continue  # already appended in a previous run
        if skip_path.exists():
            reason = skip_path.read_text(encoding="utf-8").strip() or "unspecified"
            row = {"id": int(article_id), "skip_reason": reason}
            done_names = [f"{article_id}.skip.done"]
            skipped.append(article_id)
        elif user_path.exists() and assistant_path.exists():
            user = read_pair(directory, user_path)
            assistant = read_pair(directory, assistant_path)
            if not user or not assistant:
                print(f"{article_id}: user/assistant text empty — left for retry")
                continue
            row = {"id": int(article_id), "user": user, "assistant": assistant}
            done_names = [f"{article_id}.user.done", f"{article_id}.assistant.done"]
            appended.append(article_id)
        else:
            print(f"{article_id}: incomplete pair — left for retry")
            continue
        with open(out_path, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for name in done_names:
            (directory / name).write_text("done", encoding="utf-8")
    print(f"appended {len(appended)} row(s): {appended}; "
          f"{len(skipped)} skip(s): {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--id", type=int)
    parser.add_argument("--user-file")
    parser.add_argument("--assistant-file")
    parser.add_argument("--skip")
    parser.add_argument("--scan")
    args = parser.parse_args()

    if args.scan:
        scan(args.scan, args.out)
        return
    if args.skip is not None:
        row = {"id": args.id, "skip_reason": args.skip}
    else:
        if not (args.user_file and args.assistant_file):
            sys.exit("--user-file and --assistant-file required (or --skip / --scan)")
        user = Path(args.user_file).read_text(encoding="utf-8").strip()
        assistant = Path(args.assistant_file).read_text(encoding="utf-8").strip()
        if not user or not assistant:
            sys.exit("user/assistant text is empty — not appended")
        row = {"id": args.id, "user": user, "assistant": assistant}
    with open(args.out, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"appended id={args.id} ({'skip' if args.skip else 'row'})")


if __name__ == "__main__":
    main()
