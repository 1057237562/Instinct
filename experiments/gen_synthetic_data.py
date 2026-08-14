"""
Generate a small synthetic Chinese template dataset for verifying the
Dynamic Looping + Reward-Driven Early Exit mechanism (paper §4.1 style:
repeated Chinese text on ML/NLP topics, learnable statistical patterns).

Output: experiments/data/synthetic_pretrain.jsonl  (one {"text": ...} per line)
"""
import json, random, os

random.seed(42)

SUBJECTS = [
    "人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理",
    "计算机视觉", "大语言模型", "强化学习", "Transformer", "注意力机制",
]
VERBS = [
    "是一种", "被广泛应用于", "在当代科技中扮演重要角色", "能够有效处理",
    "正在快速发展", "是人工智能领域的核心技术", "依赖于大量数据", "显著提升了",
]
OBJECTS = [
    "图像识别", "语音识别", "文本生成", "机器翻译", "推荐系统",
    "自动驾驶", "医疗诊断", "金融风控", "智能问答", "代码生成",
]
ADVERBS = ["通常", "近年来", "在实际应用中", "与传统方法相比", "在工业界"]


def make_sentence(rng: random.Random) -> str:
    subj = rng.choice(SUBJECTS)
    verb = rng.choice(VERBS)
    if "应用于" in verb or "处理" in verb or "提升了" in verb:
        obj = rng.choice(OBJECTS)
        tail = f" {obj}。"
    else:
        tail = "。"
    adv = rng.choice(ADVERBS) if rng.random() < 0.5 else ""
    return f"{adv}{subj}{verb}{tail}"


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_pretrain.jsonl")

    rng = random.Random(123)
    # 600 distinct template-composed sentences (rich but learnable statistics)
    seen = set()
    texts = []
    while len(texts) < 600:
        s = make_sentence(rng)
        if s in seen:
            continue
        seen.add(s)
        texts.append(s)

    with open(out_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    # token stats (MiniMind tokenizer, ~1.5-1.7 chars/token for Chinese)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
    lens = [len(tok.encode(t)) for t in texts]
    avg_chars = sum(len(t) for t in texts) / len(texts)
    print(f"written: {out_path}")
    print(f"samples: {len(texts)}, avg chars: {avg_chars:.1f}, "
          f"token len min/avg/max: {min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}")
    print(f"recommended max_seq_len: {max(lens) + 16} (pad budget)")
    # show 3 samples
    for t in texts[:3]:
        print("  e.g.", t)


if __name__ == "__main__":
    main()
