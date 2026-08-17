"""
Lambda sweep: does the depth-reward weight lambda=0.1 over-penalize looping,
causing premature exit + undertrained deep loop layers?

For each lambda in {0, 0.01, 0.05, 0.1, 0.5, 1.0} (2 seeds):
  - train Looped Instinct 80 iters (same data/config as before)
  - record dynamic exit steps / PPL / acc
  - record forced-depth curve k=1,2,4,8 (does extra looping help at this lambda?)
  - record avg steps per training iter (early-exit speed)

Outputs: experiments/results/lambda_sweep.json + lambda_*.png
"""
import json, math, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_instinct import InstinctConfig
from model.model_looped_instinct import LoopedInstinctConfig, LoopedInstinctForCausalLM
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1]
LAMBDAS = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
MAX_SEQ_LEN = 33
BATCH = 8
ITERS = 80
LR = 2e-4
CAP = 10
Q_THRESHOLD = 0.75
N_SUPERVISION = 5
BETA = 0.5

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


def train_track(model, ids, seed):
    """Train; return avg-steps per iter to see early-exit speed."""
    rng = random.Random(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps_curve = []
    for it in range(ITERS):
        model.train()
        x = make_batch(ids, BATCH, rng)
        opt.zero_grad()
        out = model(x, labels=x, flash_attn=False)
        out.loss.backward()
        opt.step()
        if it in [0, 5, 10, 20, 40, 60, 79]:
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


def main():
    print("=" * 72)
    print(f"Lambda sweep: {LAMBDAS} x seeds={SEEDS} x {ITERS} iters")
    print("=" * 72)
    tok, ids = load_data()

    results = {}
    for lam in LAMBDAS:
        dyn_ppl, dyn_acc, dyn_steps, curve, steps_curve = [], [], [], {}, []
        for seed in SEEDS:
            print(f"\n--- lambda={lam}, seed={seed} ---")
            m = build_looped(lam)
            sc = train_track(m, ids, seed)
            steps_curve.append(sc)
            p, a, s = eval_dynamic(m, ids)
            dyn_ppl.append(p); dyn_acc.append(a); dyn_steps.append(s)
            print(f"  dynamic: ppl={p:.4f} acc={a:.4f} steps={s:.2f}")
            for k in [1, 2, 4, 8]:
                pk, ak = eval_forced(m, ids, k)
                curve.setdefault(k, []).append(pk)
                print(f"    forced k={k}: ppl={pk:.4f} acc={ak:.4f}")
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        results[str(lam)] = {
            "dyn_ppl_mean": float(np.mean(dyn_ppl)),
            "dyn_acc_mean": float(np.mean(dyn_acc)),
            "dyn_steps_mean": float(np.mean(dyn_steps)),
            "dyn_steps_std": float(np.std(dyn_steps)),
            "forced_ppl": {str(k): float(np.mean(v)) for k, v in curve.items()},
            "steps_curve": steps_curve,
        }
        print(f"  => lambda={lam}: dyn_ppl={np.mean(dyn_ppl):.4f} "
              f"acc={np.mean(dyn_acc):.4f} steps={np.mean(dyn_steps):.2f}±{np.std(dyn_steps):.2f}")

    with open(os.path.join(RESULT_DIR, "lambda_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(f"{'λ':>7} | {'dyn PPL':>8} | {'dyn acc':>8} | {'dyn steps':>10} | {'PPL@k=1':>8} | {'PPL@k=8':>8} | Δ(k1→k8)")
    for lam in LAMBDAS:
        r = results[str(lam)]
        p1 = r["forced_ppl"]["1"]; p8 = r["forced_ppl"]["8"]
        print(f"{lam:7.2f} | {r['dyn_ppl_mean']:8.4f} | {r['dyn_acc_mean']:8.4f} | "
              f"{r['dyn_steps_mean']:5.2f}±{r['dyn_steps_std']:.2f} | {p1:8.4f} | {p8:8.4f} | {p8-p1:+.4f}")
    print(f"\nsaved → {RESULT_DIR}/lambda_sweep.json")


if __name__ == "__main__":
    main()
