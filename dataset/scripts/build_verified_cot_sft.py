"""Build concise bilingual multi-step CoT SFT with fully recomputable labels."""
from collections import Counter
from fractions import Fraction
import ast
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import orjson
from datasets import load_dataset  # noqa: F401  # before transformers/torch on Windows
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from dataset.lm_dataset import _create_chat_prompt

OUT = ROOT / "dataset" / "verified_cot_sft"
SEED = 2026091523
PER_FAMILY = 800
FAMILIES = (
    "inventory_flow", "entity_chain", "discount_shipping", "unit_rate_batches",
    "average_speed", "time_budget", "ticket_revenue", "rectangle_tiles",
    "ratio_split", "percent_then_change",
)


def canon(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def sha(value): return hashlib.sha256((value if isinstance(value, str) else canon(value)).encode()).hexdigest()


def safe_number(node):
    if isinstance(node, ast.Expression): return safe_number(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float): return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = safe_number(node.operand); return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = safe_number(node.left), safe_number(node.right)
        return left + right if isinstance(node.op, ast.Add) else left - right if isinstance(node.op, ast.Sub) else left * right if isinstance(node.op, ast.Mult) else left / right
    raise ValueError(ast.dump(node))


def verify_equations(answer):
    equations = re.findall(r"`([^`=]+=[^`]+)`", answer)
    assert len(equations) >= 2
    for equation in equations:
        parts = [part.strip() for part in equation.split("=")]
        values = [safe_number(ast.parse(part, mode="eval")) for part in parts]
        assert all(value == values[0] for value in values[1:]), equation


def make_problem(kind, rng):
    if kind == "inventory_flow":
        start = rng.randint(40, 300); sold = rng.randint(5, start // 3); damaged = rng.randint(0, 12); restock = rng.randint(5, 80)
        p = dict(start=start, sold=sold, damaged=damaged, restock=restock)
        mid = start - sold - damaged; final = mid + restock
        qzh = f"仓库开始有{start}件货物，卖出{sold}件，损坏{damaged}件，随后补货{restock}件。现在有多少件？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"A warehouse starts with {start} items, sells {sold}, loses {damaged} damaged items, then receives {restock}. How many are there now? Show the steps and end with '#### <number>'."
        steps = [(f"先扣除卖出和损坏的货物", f"First remove the sold and damaged items", f"{start}-{sold}-{damaged}={mid}"),
                 (f"再加入补货", f"Then add the new shipment", f"{mid}+{restock}={final}")]
    elif kind == "entity_chain":
        a = rng.randint(4, 60); more = rng.randint(2, 30); fewer = rng.randint(1, a + more - 1)
        b = a + more; c = b - fewer; final = a + b + c
        p = dict(a=a, more=more, fewer=fewer)
        qzh = f"甲有{a}枚徽章，乙比甲多{more}枚，丙比乙少{fewer}枚。三人共有多少枚？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"A has {a} badges. B has {more} more than A, and C has {fewer} fewer than B. How many badges do they have altogether? Show the steps and end with '#### <number>'."
        steps = [("先求乙的数量", "First find B's amount", f"{a}+{more}={b}"),
                 ("再求丙的数量", "Then find C's amount", f"{b}-{fewer}={c}"),
                 ("最后把三人的数量相加", "Finally add all three amounts", f"{a}+{b}+{c}={final}")]
    elif kind == "discount_shipping":
        price = rng.randint(2, 30) * 20; count = rng.randint(2, 9); discount = rng.choice((10, 20, 25, 50)); shipping = rng.randint(1, 30)
        subtotal = price * count; reduction = subtotal * discount // 100; final = subtotal - reduction + shipping
        p = dict(price=price, count=count, discount=discount, shipping=shipping)
        qzh = f"每件商品{price}元，购买{count}件。商品小计打{100-discount}折（即减免{discount}%），再加不参与折扣的运费{shipping}元。应付多少元？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"Each item costs {price} dollars and {count} are bought. Reduce the merchandise subtotal by {discount}%, then add an undiscounted shipping fee of {shipping} dollars. What is paid? Show the steps and end with '#### <number>'."
        steps = [("先算商品小计", "First calculate the merchandise subtotal", f"{price}*{count}={subtotal}"),
                 ("再算减免金额", "Then calculate the discount amount", f"{subtotal}*{discount}/100={reduction}"),
                 ("从小计扣除减免并加入运费", "Subtract the discount and add shipping", f"{subtotal}-{reduction}+{shipping}={final}")]
    elif kind == "unit_rate_batches":
        per_box = rng.randint(3, 18); boxes = rng.randint(2, 15); extra = rng.randint(0, 20); groups = rng.randint(2, 8)
        total = per_box * boxes + extra; final = total * groups
        p = dict(per_box=per_box, boxes=boxes, extra=extra, groups=groups)
        qzh = f"每组包含{boxes}盒，每盒{per_box}个零件，另外每组再放{extra}个散装零件。{groups}组共有多少个零件？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"Each batch has {boxes} boxes with {per_box} parts per box, plus {extra} loose parts. How many parts are in {groups} batches? Show the steps and end with '#### <number>'."
        steps = [("先求每组盒装零件数", "First find boxed parts per batch", f"{per_box}*{boxes}={per_box*boxes}"),
                 ("加入每组的散装零件", "Add the loose parts per batch", f"{per_box*boxes}+{extra}={total}"),
                 ("乘以组数", "Multiply by the number of batches", f"{total}*{groups}={final}")]
    elif kind == "average_speed":
        t1 = rng.randint(1, 5); t2 = rng.randint(1, 5); speed = rng.randint(20, 100); adjust = rng.randint(0, 10)
        d1 = speed * t1 + adjust * t2; d2 = (speed - adjust) * t2; total_d = d1 + d2; total_t = t1 + t2; final = total_d // total_t
        assert total_d % total_t == 0
        p = dict(d1=d1, t1=t1, d2=d2, t2=t2)
        qzh = f"一辆车先用{t1}小时行驶{d1}千米，又用{t2}小时行驶{d2}千米。全程平均速度是多少千米/小时？请用总路程除以总时间，逐步说明，并在最后一行写“#### 数字”。"
        qen = f"A vehicle travels {d1} km in {t1} hours, then {d2} km in {t2} hours. What is the average speed for the whole trip? Use total distance divided by total time, show the steps, and end with '#### <number>'."
        steps = [("先求总路程", "First find total distance", f"{d1}+{d2}={total_d}"),
                 ("再求总时间", "Then find total time", f"{t1}+{t2}={total_t}"),
                 ("用总路程除以总时间", "Divide total distance by total time", f"{total_d}/{total_t}={final}")]
    elif kind == "time_budget":
        breaks = rng.randint(0, 6); break_minutes = rng.choice((5, 7, 10, 12, 15, 20)); task_minutes = rng.choice((5, 6, 8, 10, 12, 15, 20, 25)); final = rng.randint(5, 30)
        usable = final * task_minutes; total = usable + breaks * break_minutes; hours, remainder = divmod(total, 60)
        p = dict(hours=hours, remainder=remainder, breaks=breaks, break_minutes=break_minutes, task_minutes=task_minutes)
        qzh = f"可用时间为{hours}小时{remainder}分钟，其中安排{breaks}次休息，每次{break_minutes}分钟。每项任务需{task_minutes}分钟，剩余工作时间最多能完成多少项？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"There are {hours} hours {remainder} minutes available, including {breaks} breaks of {break_minutes} minutes each. Each task takes {task_minutes} minutes. How many tasks fit in the remaining work time? Show the steps and end with '#### <number>'."
        total = hours * 60 + remainder; rest = breaks * break_minutes; usable = total - rest; final = usable // task_minutes
        steps = [("把总时间换成分钟", "Convert total time to minutes", f"{hours}*60+{remainder}={total}"),
                 ("计算休息总时长", "Find total break time", f"{breaks}*{break_minutes}={rest}"),
                 ("扣除休息后除以每项用时", "Subtract breaks and divide by time per task", f"({total}-{rest})/{task_minutes}={final}")]
    elif kind == "ticket_revenue":
        adult_n = rng.randint(10, 80); child_n = rng.randint(5, 60); adult_price = rng.randint(5, 25); child_price = rng.randint(2, adult_price - 1); cost = rng.randint(10, 100)
        adult_total = adult_n * adult_price; child_total = child_n * child_price; gross = adult_total + child_total; final = gross - cost
        p = dict(adult_n=adult_n, child_n=child_n, adult_price=adult_price, child_price=child_price, cost=cost)
        qzh = f"售出成人票{adult_n}张，每张{adult_price}元；儿童票{child_n}张，每张{child_price}元。扣除{cost}元场地费后，净收入是多少元？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"A venue sells {adult_n} adult tickets at {adult_price} dollars and {child_n} child tickets at {child_price} dollars. After a venue cost of {cost} dollars, what is net revenue? Show the steps and end with '#### <number>'."
        steps = [("成人票收入", "Adult-ticket revenue", f"{adult_n}*{adult_price}={adult_total}"),
                 ("儿童票收入", "Child-ticket revenue", f"{child_n}*{child_price}={child_total}"),
                 ("合计后扣除场地费", "Add revenue and subtract the venue cost", f"{adult_total}+{child_total}-{cost}={final}")]
    elif kind == "rectangle_tiles":
        length = rng.randint(3, 20); width = rng.randint(3, 20); tile = rng.choice((1, 2, 4)); length *= tile; width *= tile; excluded = rng.randint(0, (length*width)//(tile*tile)//4)
        area = length * width; tile_area = tile * tile; all_tiles = area // tile_area; final = all_tiles - excluded
        p = dict(length=length, width=width, tile=tile, excluded=excluded)
        qzh = f"一个长{length}米、宽{width}米的矩形地面用边长{tile}米的正方形地砖铺设，其中{excluded}块位置留空。实际需要多少块地砖？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"A {length} m by {width} m rectangular floor uses square tiles of side {tile} m, with {excluded} tile positions left empty. How many tiles are actually needed? Show the steps and end with '#### <number>'."
        steps = [("先求地面面积", "First find floor area", f"{length}*{width}={area}"),
                 ("求每块地砖面积", "Find one tile's area", f"{tile}*{tile}={tile_area}"),
                 ("用总面积除以单块面积，再扣除留空位置", "Divide by tile area and remove empty positions", f"{area}/{tile_area}-{excluded}={final}")]
    elif kind == "ratio_split":
        left = rng.randint(1, 12); right = rng.randint(1, 12); unit = rng.randint(3, 60); total = (left + right) * unit; final = left * unit
        p = dict(total=total, left=left, right=right)
        qzh = f"把{total}个物品按{left}:{right}分给甲、乙。甲得到多少个？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"Split {total} items between A and B in the ratio {left}:{right}. How many does A receive? Show the steps and end with '#### <number>'."
        steps = [("先求总份数", "First find the total number of ratio parts", f"{left}+{right}={left+right}"),
                 ("求每份数量", "Find the amount per part", f"{total}/{left+right}={unit}"),
                 ("甲占对应份数", "Multiply by A's number of parts", f"{unit}*{left}={final}")]
    else:
        base = rng.randint(2, 30) * 20; percent = rng.choice((10, 20, 25, 50)); added = rng.randint(1, 40)
        part = base * percent // 100; final = part + added
        p = dict(base=base, percent=percent, added=added)
        qzh = f"先取{base}的{percent}%，再把结果增加{added}。最终是多少？请逐步说明，并在最后一行写“#### 数字”。"
        qen = f"First take {percent}% of {base}, then increase that result by {added}. What is the final value? Show the steps and end with '#### <number>'."
        steps = [("把百分数对应到原数", "Apply the percentage to the original value", f"{base}*{percent}/100={part}"),
                 ("再增加指定数量", "Then add the specified amount", f"{part}+{added}={final}")]
    return p, qzh, qen, steps, final


def render_answer(steps, final, zh, style):
    lines = []
    for index, (zh_text, en_text, equation) in enumerate(steps, 1):
        text = zh_text if zh else en_text
        if style == 0:
            prefix = f"第{index}步" if zh else f"{index}."
        elif style == 1:
            prefix = ("先" if index == 1 else "然后") if zh else ("First" if index == 1 else "Then")
        else:
            prefix = (("首先" if index == 1 else "最后" if index == len(steps) else "接着") if zh else
                      ("Start by" if index == 1 else "Finally" if index == len(steps) else "Next"))
        lines.append(f"{prefix} {text}：`{equation}`" if zh else f"{prefix} {text}: `{equation}`")
    lines.append((f"所以最终结果是{final}。" if zh else f"Therefore, the final result is {final}."))
    lines.append(f"#### {final}")
    answer = "\n".join(lines)
    verify_equations(answer)
    assert answer.splitlines()[-1] == f"#### {final}"
    return answer


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_semantics = set()
    skipped_duplicate_groups = 0
    for family_index, family in enumerate(FAMILIES):
        for i in range(PER_FAMILY):
            rng = random.Random(SEED + family_index * 100_000 + i)
            params, qzh, qen, steps, final = make_problem(family, rng)
            semantic_hash = sha({"family": family, "parameters": params})
            if semantic_hash in seen_semantics:
                skipped_duplicate_groups += 1
                continue
            seen_semantics.add(semantic_hash)
            split = "validation" if int(semantic_hash[:8], 16) % 10 == 0 else "train"
            style = 2 if split == "validation" else int(semantic_hash[8:16], 16) % 2
            for zh, question in ((True, qzh), (False, qen)):
                answer = render_answer(steps, final, zh, style)
                rows.append({"family": family, "parameters": params, "final": final,
                             "language": "zh" if zh else "en", "style": style,
                             "semantic_hash": semantic_hash, "split": split,
                             "conversations": [{"role": "user", "content": question},
                                               {"role": "assistant", "content": answer}]})
    assert len({sha(r["conversations"][0]["content"]) for r in rows}) == len(rows)
    assert not ({r["semantic_hash"] for r in rows if r["split"] == "train"} &
                {r["semantic_hash"] for r in rows if r["split"] == "validation"})
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    for item in rows:
        item["tokens"] = len(tokenizer.backend_tokenizer.encode(
            _create_chat_prompt(tokenizer, item["conversations"]), add_special_tokens=False).ids)
        assert item["tokens"] <= 768
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    random.Random(SEED).shuffle(train); random.Random(SEED + 1).shuffle(validation)
    manifests = {}
    with gzip.open(OUT / "provenance.jsonl.gz", "wt", encoding="utf-8", newline="\n") as meta:
        for split, data in (("train", train), ("validation", validation)):
            digest = hashlib.sha256()
            with (OUT / f"{split}.jsonl").open("wb") as f:
                for index, item in enumerate(data):
                    verify_equations(item["conversations"][-1]["content"])
                    line = orjson.dumps({"conversations": item["conversations"]}) + b"\n"
                    f.write(line); digest.update(line)
                    meta.write(json.dumps({**{k: v for k, v in item.items() if k != "conversations"},
                                           "file_split": split, "row": index,
                                           "question": item["conversations"][0]["content"],
                                           "answer": item["conversations"][1]["content"]}, ensure_ascii=False) + "\n")
            manifests[split] = {"rows": len(data), "tokens": sum(r["tokens"] for r in data), "sha256": digest.hexdigest()}
    report = {"seed": SEED, "families": list(FAMILIES), "files": manifests,
              "train_family_counts": dict(Counter(r["family"] for r in train)),
              "validation_family_counts": dict(Counter(r["family"] for r in validation)),
              "skipped_duplicate_semantic_groups": skipped_duplicate_groups,
              "max_tokens": max(r["tokens"] for r in rows), "public_teacher_rows": 0,
              "verification": "Every displayed equation and final answer checked; independent audit is separate."}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
