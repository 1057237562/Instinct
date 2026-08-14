"""Generate verification plots from experiments/results/curves.npz + summary.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
d = np.load(os.path.join(RESULT_DIR, "curves.npz"), allow_pickle=True)
s = json.load(open(os.path.join(RESULT_DIR, "summary.json"), encoding="utf-8"))
iters = d["iters"]

std_loss = np.array(d["std_loss"]).mean(0)
loop_loss = np.array(d["loop_loss"]).mean(0)
loop_steps = np.array(d["loop_steps"]).mean(0)
std_acc = np.array(d["std_acc"]).mean(0)
loop_acc = np.array(d["loop_acc"]).mean(0)

# ── fig 1: total loss curves ──
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
ax.plot(iters, std_loss, label="Standard MiniMind (8 layers, fixed)", lw=2)
ax.plot(iters, loop_loss, label="Looped MiniMind (dynamic loop + reward)", lw=2)
ax.set_xlabel("Training iteration"); ax.set_ylabel("Total loss")
ax.set_title("Training Loss: Standard vs Looped (mean over 5 seeds)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULT_DIR, "loss_curves.png"), dpi=150)
plt.close(fig)

# ── fig 2: avg loop steps + loss (paper §4.2 style) ──
fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
ax1.plot(iters, loop_steps, "o-", color="tab:red", lw=2, label="Avg loop steps")
ax1.set_xlabel("Training iteration"); ax1.set_ylabel("Avg loop steps", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax2 = ax1.twinx()
ax2.plot(iters, loop_loss, "s-", color="tab:blue", lw=1.5, label="Looped total loss")
ax2.set_ylabel("Total loss", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_title("Emergent Early Exit: loop steps decrease with training")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
ax1.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULT_DIR, "steps_curve.png"), dpi=150)
plt.close(fig)

# ── fig 3: accuracy ──
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(iters, std_acc, label="Standard", lw=2)
ax.plot(iters, loop_acc, label="Looped", lw=2)
ax.set_xlabel("Training iteration"); ax.set_ylabel("Token accuracy")
ax.set_title("Token Accuracy: Standard vs Looped (mean over 5 seeds)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULT_DIR, "acc_curves.png"), dpi=150)
plt.close(fig)

print("plots written:")
for f in ("loss_curves.png", "steps_curve.png", "acc_curves.png"):
    p = os.path.join(RESULT_DIR, f)
    print("  ", p, os.path.getsize(p), "bytes")
