"""
Isolate the impact of the Looped structure on model accuracy and PPL.

Three models, same pretrained start (out/pretrain_768.pth), same data, same
training config (3 seeds x 80 iters, lr 2e-4, AdamW):

  A) Standard MiniMind (8 fixed layers)                     — quality baseline
  B) Looped MiniMind, depth_reward=0.1 (paper default)      — loop + reward
  C) Looped MiniMind, depth_reward=0.0 (ablation)           — pure loop, no reward

Measurements:
  1. Same-depth comparison: Standard(8 layers) vs Looped forced to k=1 step
     (encoder 2 + 1*loop 3 + output 3 = 8 effective layers).
  2. Loop-depth scaling: Looped forced to k = 1,2,4,8 loop steps → PPL/acc
     (test-time compute curve; does more looping improve quality?).
  3. Dynamic-exit quality: Looped dynamic (q_th=0.75) vs forced-full-depth.
  4. Difficulty split: easy/medium/hard samples → steps used & quality.

Outputs: experiments/results/impact_*.png, impact_summary.json
"""
import json, math, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_looped_minimind import LoopedMiniMindConfig, LoopedMiniMindForCausalLM
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
MAX_SEQ_LEN = 33
BATCH = 8
ITERS = 80
LR = 2e-4
CAP = 10
Q_THRESHOLD = 0.75
N_SUPERVISION = 5
BETA = 0.5
DEPTH_REWARD = 0.1

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    tok = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "synthetic_pretrain.jsonl")
    texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    ids = [tok.encode(t)[:MAX_SEQ_LEN] for t in texts]
    return tok, ids


def make_batch(ids, batch_size, rng):
    seqs = [rng.choice(ids) for _ in range(batch_size)]
    max_len = max(len(s) for s in seqs)
    inp = torch.full((batch_size, max_len), 0, dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s)
    return inp.to(DEVICE)


def build_standard():
    cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, vocab_size=6400,
                         flash_attn=False, inference_rope_scaling=False)
    model = MiniMindForCausalLM(cfg).to(DEVICE)
    sd = torch.load("out/pretrain_768.pth", map_location=DEVICE)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    return model


def build_looped(depth_reward):
    cfg = LoopedMiniMindConfig(
        hidden_size=768, num_hidden_layers=8, vocab_size=6400,
        flash_attn=False, inference_rope_scaling=False,
        loop_encoder_layers=[0, 1], loop_body_layers=[2, 3, 4],
        loop_output_layers=[5, 6, 7], loop_max_steps=CAP,
        q_threshold=Q_THRESHOLD, exit_in_training=True,
        n_supervision=N_SUPERVISION, beta=BETA, depth_reward=depth_reward,
    )
    model = LoopedMiniMindForCausalLM(cfg).to(DEVICE)
    sd = torch.load("out/pretrain_768.pth", map_location=DEVICE)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_pretrained_weights(sd)
    with torch.no_grad():
        model.model.q_head[1].bias.fill_(-3.0)
    return model


def train(model, ids, seed, is_looped):
    rng = random.Random(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for it in range(ITERS):
        model.train()
        x = make_batch(ids, BATCH, rng)
        opt.zero_grad()
        out = model(x, labels=x, flash_attn=False)
        out.loss.backward()
        opt.step()


def eval_ppl_acc_from_logits(logits, y):
    ll = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                         y.reshape(-1), reduction="sum")
    acc = (logits[..., :-1, :].argmax(-1) == y).float().mean().item()
    return float(ll), y.numel(), acc


def collect(model, ids, batch_draws=20):
    """Collect all (tokens, log-lik, acc) over fixed draws; returns aggregates."""
    rng = random.Random(1234)
    tot_ll, tot_tok, accs = 0.0, 0, []
    with torch.no_grad():
        for _ in range(batch_draws):
            x = make_batch(ids, BATCH, rng)
            logits = model.lm_head(model.model(x, training=False)[0])
            ll, n, acc = eval_ppl_acc_from_logits(logits, x[..., 1:])
            tot_ll += ll; tot_tok += n; accs.append(acc)
    return math.exp(tot_ll / tot_tok), float(np.mean(accs))


def collect_forced(model, ids, k, batch_draws=20):
    """Force exactly k loop steps by setting q_threshold above sigmoid range."""
    model.model.config.q_threshold = 1.5
    old_cap = model.model.config.loop_max_steps
    model.model.config.loop_max_steps = k
    rng = random.Random(1234)
    tot_ll, tot_tok, accs, steps = 0.0, 0, [], []
    with torch.no_grad():
        for _ in range(batch_draws):
            x = make_batch(ids, BATCH, rng)
            hidden, _, _, info = model.model(x, training=False)
            steps.append(info["avg_steps"])
            logits = model.lm_head(hidden)
            ll, n, acc = eval_ppl_acc_from_logits(logits, x[..., 1:])
            tot_ll += ll; tot_tok += n; accs.append(acc)
    model.model.config.loop_max_steps = old_cap
    model.model.config.q_threshold = Q_THRESHOLD
    return math.exp(tot_ll / tot_tok), float(np.mean(accs)), float(np.mean(steps))


def collect_dynamic(model, ids, batch_draws=20):
    model.model.config.q_threshold = Q_THRESHOLD
    model.model.config.loop_max_steps = CAP
    rng = random.Random(1234)
    tot_ll, tot_tok, accs, steps = 0.0, 0, [], []
    with torch.no_grad():
        for _ in range(batch_draws):
            x = make_batch(ids, BATCH, rng)
            hidden, _, _, info = model.model(x, training=False)
            steps.append(info["avg_steps"])
            logits = model.lm_head(hidden)
            ll, n, acc = eval_ppl_acc_from_logits(logits, x[..., 1:])
            tot_ll += ll; tot_tok += n; accs.append(acc)
    return math.exp(tot_ll / tot_tok), float(np.mean(accs)), float(np.mean(steps))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("Isolating Loop impact on accuracy & PPL "
          f"(seeds={SEEDS}, iters={ITERS}, lr={LR})")
    print("=" * 72)
    tok, ids = load_data()

    # difficulty buckets by token length (harder = longer template)
    lens = np.array([len(s) for s in ids])
    p33, p66 = np.percentile(lens, [33, 66])
    buckets = {
        "easy(<=p33)": [i for i, l in enumerate(lens) if l <= p33],
        "medium": [i for i, l in enumerate(lens) if p33 < l <= p66],
        "hard(>p66)": [i for i, l in enumerate(lens) if l > p66],
    }
    bucket_ids = {k: [ids[i] for i in v] for k, v in buckets.items()}

    rows = {"A_standard": [], "B_looped_l0.1": [], "C_looped_l0.0": []}
    forced_curves = {"B": [], "C": []}
    dynamic_rows = {"B": [], "C": []}
    difficulty_rows = {"B": [], "C": []}

    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")

        # A: standard
        m = build_standard()
        train(m, ids, seed, is_looped=False)
        ppl, acc = collect(m, ids)
        rows["A_standard"].append({"ppl": ppl, "acc": acc})
        print(f"  [A standard]      ppl={ppl:.4f} acc={acc:.4f}")

        # B: looped λ=0.1
        mb = build_looped(0.1)
        train(mb, ids, seed, is_looped=True)
        ppl_dyn, acc_dyn, steps_dyn = collect_dynamic(mb, ids)
        dynamic_rows["B"].append({"ppl": ppl_dyn, "acc": acc_dyn, "steps": steps_dyn})
        curve_b = {}
        for k in [1, 2, 4, 8]:
            p, a, s = collect_forced(mb, ids, k)
            curve_b[k] = {"ppl": p, "acc": a, "steps": s}
        forced_curves["B"].append(curve_b)
        rows["B_looped_l0.1"].append({"ppl_dyn": ppl_dyn, "acc_dyn": acc_dyn,
                                      "steps_dyn": steps_dyn, "curve": curve_b})
        print(f"  [B looped λ0.1]   dyn ppl={ppl_dyn:.4f} acc={acc_dyn:.4f} "
              f"steps={steps_dyn:.2f}")
        for k in [1, 2, 4, 8]:
            print(f"      forced k={k}: ppl={curve_b[k]['ppl']:.4f} "
                  f"acc={curve_b[k]['acc']:.4f}")

        # C: looped λ=0.0 (ablation)
        mc = build_looped(0.0)
        train(mc, ids, seed, is_looped=True)
        ppl_dyn, acc_dyn, steps_dyn = collect_dynamic(mc, ids)
        dynamic_rows["C"].append({"ppl": ppl_dyn, "acc": acc_dyn, "steps": steps_dyn})
        curve_c = {}
        for k in [1, 2, 4, 8]:
            p, a, s = collect_forced(mc, ids, k)
            curve_c[k] = {"ppl": p, "acc": a, "steps": s}
        forced_curves["C"].append(curve_c)
        rows["C_looped_l0.0"].append({"ppl_dyn": ppl_dyn, "acc_dyn": acc_dyn,
                                      "steps_dyn": steps_dyn, "curve": curve_c})
        print(f"  [C looped λ0.0]   dyn ppl={ppl_dyn:.4f} acc={acc_dyn:.4f} "
              f"steps={steps_dyn:.2f}")

        # difficulty split on model B
        db = {}
        for name, bid in bucket_ids.items():
            p, a, s = collect_dynamic(mb, bid)
            db[name] = {"ppl": p, "acc": a, "steps": s}
            print(f"      difficulty {name}: ppl={p:.4f} acc={a:.4f} steps={s:.2f}")
        difficulty_rows["B"].append(db)

        del m, mb, mc
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ── aggregate ──
    def mean_std(vals, key):
        arr = np.array([v[key] for v in vals])
        return arr.mean(), arr.std()

    print("\n" + "=" * 72)
    print("AGGREGATE (mean ± std over 3 seeds)")
    print("=" * 72)

    m_std_ppl, s_std_ppl = mean_std(rows["A_standard"], "ppl")
    m_std_acc, s_std_acc = mean_std(rows["A_standard"], "acc")
    print(f"A standard       : ppl {m_std_ppl:.4f}±{s_std_ppl:.4f}  acc {m_std_acc:.4f}±{s_std_acc:.4f}")

    for tag, key in [("B λ0.1", "B_looped_l0.1"), ("C λ0.0", "C_looped_l0.0")]:
        m_p, s_p = mean_std(rows[key], "ppl_dyn")
        m_a, s_a = mean_std(rows[key], "acc_dyn")
        m_s, s_s = mean_std(rows[key], "steps_dyn")
        print(f"{tag} dynamic    : ppl {m_p:.4f}±{s_p:.4f}  acc {m_a:.4f}±{s_a:.4f}  steps {m_s:.2f}±{s_s:.2f}")

    print("\nforced-depth curve (ppl / acc), mean over seeds:")
    for tag, curve in [("B λ0.1", forced_curves["B"]), ("C λ0.0", forced_curves["C"])]:
        for k in [1, 2, 4, 8]:
            vals = [c[k] for c in curve]
            p = np.mean([v["ppl"] for v in vals]); a = np.mean([v["acc"] for v in vals])
            print(f"  {tag} k={k}: ppl {p:.4f}  acc {a:.4f}")

    print("\ndifficulty split (model B, dynamic):")
    for name in bucket_ids:
        vals = [d[name] for d in difficulty_rows["B"]]
        p = np.mean([v["ppl"] for v in vals]); a = np.mean([v["acc"] for v in vals])
        s = np.mean([v["steps"] for v in vals])
        print(f"  {name:14s}: ppl {p:.4f}  acc {a:.4f}  steps {s:.2f}")

    summary = {
        "standard": {"ppl_mean": m_std_ppl, "ppl_std": s_std_ppl,
                     "acc_mean": m_std_acc, "acc_std": s_std_acc},
        "looped_l0.1_dynamic": {"ppl_mean": mean_std(rows["B_looped_l0.1"], "ppl_dyn")[0],
                                "acc_mean": mean_std(rows["B_looped_l0.1"], "acc_dyn")[0],
                                "steps_mean": mean_std(rows["B_looped_l0.1"], "steps_dyn")[0]},
        "looped_l0.0_dynamic": {"ppl_mean": mean_std(rows["C_looped_l0.0"], "ppl_dyn")[0],
                                "acc_mean": mean_std(rows["C_looped_l0.0"], "acc_dyn")[0],
                                "steps_mean": mean_std(rows["C_looped_l0.0"], "steps_dyn")[0]},
        "forced_curves": {
            "B": {str(k): {"ppl": np.mean([c[k]["ppl"] for c in forced_curves["B"]]),
                            "acc": np.mean([c[k]["acc"] for c in forced_curves["B"]])}
                  for k in [1, 2, 4, 8]},
            "C": {str(k): {"ppl": np.mean([c[k]["ppl"] for c in forced_curves["C"]]),
                            "acc": np.mean([c[k]["acc"] for c in forced_curves["C"]])}
                  for k in [1, 2, 4, 8]},
        },
        "difficulty": {name: {"ppl": np.mean([d[name]["ppl"] for d in difficulty_rows["B"]]),
                              "acc": np.mean([d[name]["acc"] for d in difficulty_rows["B"]]),
                              "steps": np.mean([d[name]["steps"] for d in difficulty_rows["B"]])}
                       for name in bucket_ids},
    }
    with open(os.path.join(RESULT_DIR, "impact_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsaved → {RESULT_DIR}/impact_summary.json")


if __name__ == "__main__":
    main()
