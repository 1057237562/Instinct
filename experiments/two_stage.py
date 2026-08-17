"""
Two-stage hypothesis test: should early-exit training be deferred to SFT?

The observation from lambda_sweep: even with lambda=0, forcing more loop
steps does NOT improve PPL — the shared 3-layer loop body learns a "1-step
shortcut" and never learns to progressively refine. This suggests the problem
is NOT lambda but *undertrained deep loop layers* (samples exit at step 1, so
deeper loop states are barely visited).

Hypothesis: during a "pretrain-like" phase the loop should run FULL depth with
no exit pressure (so every loop state is trained), and only later ("SFT-like")
should early exit be trained. Three schemes, 2 seeds x 80 iters, same data:

  S1 baseline     : lambda=0.1 throughout (paper default)
  S2 two-stage-mild: 60 iters lambda=0 (natural exit) -> 20 iters lambda=0.1
  S3 two-stage-full: 60 iters lambda=0 + exit_in_training=False (FORCE all
                     samples to run full cap, deep loop states fully trained)
                     -> 20 iters lambda=0.1 with exit enabled

Key questions:
  Q1: does S3 achieve lower PPL than S1 at dynamic exit? (deep loop body pays off)
  Q2: does S3's forced-depth curve slope DOWN (more loops -> better)? If yes,
      the loop body finally learned progressive refinement.
  Q3: can S3 still exit at ~1 step dynamically (early-exit still works after
      full-depth pretraining)?

Outputs: experiments/results/two_stage.json + two_stage_*.png
"""
import json, math, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_looped_instinct import LoopedInstinctConfig, LoopedInstinctForCausalLM
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1]
PHASE1, PHASE2 = 60, 20
MAX_SEQ_LEN = 33
BATCH = 8
LR = 2e-4
CAP = 10
Q_THRESHOLD = 0.75
N_SUPERVISION = 5
BETA = 0.5
LAMBDA_SFT = 0.1

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_data():
    tok = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "synthetic_pretrain.jsonl")
    texts = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    return tok, [tok.encode(t)[:MAX_SEQ_LEN] for t in texts]


def make_batch(ids, batch_size, rng):
    seqs = [rng.choice(ids) for _ in range(batch_size)]
    max_len = max(len(s) for s in seqs)
    inp = torch.full((batch_size, max_len), 0, dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s)
    return inp.to(DEVICE)


def build_looped(depth_reward):
    cfg = LoopedInstinctConfig(
        hidden_size=768, num_hidden_layers=8, vocab_size=6400,
        flash_attn=False, inference_rope_scaling=False,
        loop_encoder_layers=[0, 1], loop_body_layers=[2, 3, 4],
        loop_output_layers=[5, 6, 7], loop_max_steps=CAP,
        q_threshold=Q_THRESHOLD, exit_in_training=True,
        n_supervision=N_SUPERVISION, beta=BETA, depth_reward=depth_reward,
    )
    model = LoopedInstinctForCausalLM(cfg).to(DEVICE)
    sd = torch.load("out/pretrain_768.pth", map_location=DEVICE)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_pretrained_weights(sd)
    with torch.no_grad():
        model.model.q_head[1].bias.fill_(-3.0)
    return model


def train_iters(model, ids, seed, n_iters, depth_reward, exit_enabled, tag):
    model.set_depth_reward(depth_reward)
    model.model.config.exit_in_training = exit_enabled
    rng = random.Random(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_curve = []
    for it in range(n_iters):
        model.train()
        x = make_batch(ids, BATCH, rng)
        opt.zero_grad()
        out = model(x, labels=x, flash_attn=False)
        out.loss.backward()
        opt.step()
        if it in [0, 5, 10, 20, 40, n_iters - 1]:
            steps_curve.append([it, float(model.last_avg_steps)])
    return steps_curve


def eval_dynamic(model, ids):
    model.model.config.q_threshold = Q_THRESHOLD
    model.model.config.loop_max_steps = CAP
    rng = random.Random(1234)
    tot_ll, tot_tok, accs, steps = 0.0, 0, [], []
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            x = make_batch(ids, BATCH, rng)
            hidden, _, _, info = model.model(x, training=False)
            steps.append(info["avg_steps"])
            logits = model.lm_head(hidden)
            ll = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                                 x[..., 1:].reshape(-1), reduction="sum")
            tot_ll += float(ll); tot_tok += x[..., 1:].numel()
            accs.append((logits[..., :-1, :].argmax(-1) == x[..., 1:]).float().mean().item())
    return math.exp(tot_ll / tot_tok), float(np.mean(accs)), float(np.mean(steps))


def eval_forced(model, ids, k):
    model.model.config.q_threshold = 1.5
    old_cap = model.model.config.loop_max_steps
    model.model.config.loop_max_steps = k
    rng = random.Random(1234)
    tot_ll, tot_tok, accs = 0.0, 0, []
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            x = make_batch(ids, BATCH, rng)
            hidden, _, _, _ = model.model(x, training=False)
            logits = model.lm_head(hidden)
            ll = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                                 x[..., 1:].reshape(-1), reduction="sum")
            tot_ll += float(ll); tot_tok += x[..., 1:].numel()
            accs.append((logits[..., :-1, :].argmax(-1) == x[..., 1:]).float().mean().item())
    model.model.config.loop_max_steps = old_cap
    model.model.config.q_threshold = Q_THRESHOLD
    return math.exp(tot_ll / tot_tok), float(np.mean(accs))


def run_scheme(name, seed):
    print(f"\n=== {name}, seed={seed} ===")
    m = build_looped(0.1 if name == "S1" else 0.0)
    if name == "S1":
        sc = train_iters(m, ids, seed, PHASE1 + PHASE2, 0.1, True, name)
    elif name == "S2":
        sc = train_iters(m, ids, seed, PHASE1, 0.0, True, name)
        sc2 = train_iters(m, ids, seed, PHASE2, LAMBDA_SFT, True, name)
        sc = sc + [[PHASE1 + p[0], p[1]] for p in sc2]
    else:  # S3: force full depth in phase 1
        sc = train_iters(m, ids, seed, PHASE1, 0.0, False, name)
        sc2 = train_iters(m, ids, seed, PHASE2, LAMBDA_SFT, True, name)
        sc = sc + [[PHASE1 + p[0], p[1]] for p in sc2]
    p, a, s = eval_dynamic(m, ids)
    print(f"  dynamic: ppl={p:.4f} acc={a:.4f} steps={s:.2f}")
    curve = {}
    for k in [1, 2, 4, 8]:
        pk, ak = eval_forced(m, ids, k)
        curve[k] = {"ppl": pk, "acc": ak}
        print(f"    forced k={k}: ppl={pk:.4f} acc={ak:.4f}")
    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"dyn_ppl": p, "dyn_acc": a, "dyn_steps": s, "curve": curve, "steps_curve": sc}


def main():
    global ids
    print("=" * 72)
    print(f"Two-stage hypothesis: {SEEDS} seeds, phase1={PHASE1}, phase2={PHASE2}")
    print("  S1 baseline λ=0.1 | S2 λ0->λ0.1 (natural) | S3 λ0+full-depth ->λ0.1")
    print("=" * 72)
    tok, ids = load_data()

    results = {}
    for name in ["S1", "S2", "S3"]:
        rows = [run_scheme(name, seed) for seed in SEEDS]
        results[name] = {
            "dyn_ppl_mean": float(np.mean([r["dyn_ppl"] for r in rows])),
            "dyn_ppl_std": float(np.std([r["dyn_ppl"] for r in rows])),
            "dyn_acc_mean": float(np.mean([r["dyn_acc"] for r in rows])),
            "dyn_steps_mean": float(np.mean([r["dyn_steps"] for r in rows])),
            "dyn_steps_std": float(np.std([r["dyn_steps"] for r in rows])),
            "forced": {str(k): {"ppl": float(np.mean([r["curve"][k]["ppl"] for r in rows])),
                                 "acc": float(np.mean([r["curve"][k]["acc"] for r in rows]))}
                       for k in [1, 2, 4, 8]},
            "steps_curve": [r["steps_curve"] for r in rows],
        }

    with open(os.path.join(RESULT_DIR, "two_stage.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("SUMMARY (mean over 2 seeds)")
    print("=" * 72)
    print(f"{'scheme':<6} | {'dyn PPL':>9} | {'dyn acc':>8} | {'dyn steps':>10} | {'PPL@k1':>7} | {'PPL@k4':>7} | {'PPL@k8':>7}")
    for name in ["S1", "S2", "S3"]:
        r = results[name]
        f = r["forced"]
        print(f"{name:<6} | {r['dyn_ppl_mean']:9.4f} | {r['dyn_acc_mean']:8.4f} | "
              f"{r['dyn_steps_mean']:5.2f}±{r['dyn_steps_std']:.2f} | "
              f"{f['1']['ppl']:7.4f} | {f['4']['ppl']:7.4f} | {f['8']['ppl']:7.4f}")
    print(f"\nsaved → {RESULT_DIR}/two_stage.json")


if __name__ == "__main__":
    main()
