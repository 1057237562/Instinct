"""Quantify Muon parameter-trajectory deviation under low-precision weights.

Purpose
-------
Training can store 2D Muon parameters directly in bfloat16/float16
(``--param_dtype bf16`` / ``fp16``). Every optimizer update is then rounded to
the storage dtype, so an update smaller than one ULP vanishes. This experiment
measures that accumulated deviation on the real ``BatchedMuon`` optimizer and
the real ``InstinctForCausalLM`` forward/backward, using a tiny dense model. It
then shows that ``use_mantissa=True`` removes the *optimizer-side* deviation
bit-exactly: the mantissa arm's internal fp32 trajectory equals the fp32
reference to the last bit.

Why shared gradients
--------------------
This codebase has a documented MoE-routing chaos effect (see
``tests/test_gram_muon_short_training.py``): discrete top-1 routing means ANY
rounding difference flips token -> expert assignment, so two arms that each run
their own forward diverge for reasons unrelated to the optimizer. In the default
``--mode optimizer`` the reference model's gradient is computed once per step,
rounded to bfloat16, and handed *identically* to every arm. The measured delta
is therefore attributable to parameter storage and optimizer arithmetic only,
never to forward-pass or routing differences. Gradients are identical across
arms by construction.

Why split-init
--------------
Arms are ``copy.deepcopy`` of the fp32 reference (identical init) and then cast
without round-to-nearest: ``_split_precise`` truncates the fp32 bits into a
bfloat16 high half plus a uint16 low half, and the arm keeps only the high half.
For the mantissa arm the low half is seeded back into the optimizer state, so
its internal fp32 value equals the original fp32 init *exactly* -- the one-time
placement offset is removed and only per-step optimizer error remains. (The
production ``model.to(bf16)`` cast is round-to-nearest and adds a one-time
placement offset; that offset is printed separately as an informational stat.)
``_split_precise`` / ``_reconstruct_precise`` are a truncating bit split, so
split-then-reconstruct is a bitwise identity; the ``internal_exact`` column
verifies that invariant every step.

Why a duplicate fp32 arm (baseline)
-----------------------------------
``fp32b`` is a second, independent fp32 arm: a ``copy.deepcopy`` of the
reference model taken before any cast, with no dtype change and no mantissa
seeding. In ``--mode optimizer`` the two fp32 arms receive identical gradients
and run identical deterministic fp32 ops, so ``fp32b`` MUST be bit-exact every
step (``bit_exact_frac == 1``) with ``accum_delta == 0``. That is the proof
that any nonzero low-precision delta is causally attributable to the storage
dtype, not to measurement noise in the harness. In ``--mode train`` each arm
runs its own forward/backward, so ``fp32b`` is no longer expected to be
bit-exact (CUDA nondeterministic reductions, amplified by trajectory chaos);
its delta is then the process noise floor against which the bf16/fp16 deltas
are judged.

What train mode adds
--------------------
``--mode train`` is end-to-end: each arm owns a deepcopy cast with the
production round-to-nearest ``.to(dtype)`` and runs its own forward, backward and
step (bf16/fp16 arms forward natively in their dtype, the reference in fp32). Its
deltas therefore include forward/backward dtype effects and compound over steps,
so the mantissa arm is *not* expected to be bit-exact there.

Memory tier: bf16-quantized momentum
------------------------------------
``bf16_mantissa_bf16mom`` is ``bf16_mantissa`` with the momentum buffer rounded
to bfloat16 after every optimizer step (in place, so the buffer object and its
fp32 arithmetic are unchanged -- only the stored values sit on the bf16 grid).
It models an 8-byte/parameter Muon tier (bf16 param + uint16 remainder + bf16
grad + bf16 momentum) against the 10-byte mantissa tier (fp32 momentum), and
isolates the loss caused solely by dropping momentum from fp32 to bf16.
``bf16_mantissa_mom24`` is the intermediate rung: the same lossless-parameter
path with momentum truncated to 24 bits (sign + 8 exponent + 15 explicit
mantissa bits = 16 significant bits; low 8 mantissa bits cleared), i.e. a
9-byte/parameter tier.

Metrics
-------
``accum_delta``/``traj_delta`` are fp64 parameter-space deviation norms,
normalized by the reference displacement, and are unchanged here. ``zero_elem``
is a per-ELEMENT fraction: among elements whose reference value changed this
step, the share whose value in the arm did NOT change -- i.e. the
quantization-vanishing rate. (An earlier per-tensor version required a whole
tensor to be frozen and so read 0.0 always; the per-element form is the
guard-rationale evidence.)

Policy note
-----------
Production additionally guards ``lr <= 1e-4`` for directly updated low-precision
parameters (``trainer_utils._validate_optimizer_param_precision``). This
experiment deliberately bypasses that policy layer by using ``BatchedMuon``
directly, so it can probe learning rates that production would reject.

Run from the repo root::

    python experiments/check_muon_bf16_delta.py
    python experiments/check_muon_bf16_delta.py --mode train --steps 100
"""

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets  # noqa: F401 -- Windows DLL import order
import trainer.compile_cache  # noqa: F401 -- before torch
import torch

from model.model_instinct import InstinctConfig, InstinctForCausalLM
from tests.helpers import make_tiny_config
from trainer.batched_muon import BatchedMuon, _reconstruct_precise, _split_precise

BATCH, SEQ, PERIOD = 8, 96, 8
EPS = 1e-30
ARM_CHOICES = ("fp32b", "bf16", "bf16_mantissa", "bf16_mantissa_bf16mom",
               "bf16_mantissa_mom24", "fp16")

# Loosely-typed record for one arm: params, optimizer, dtype policy, shadows.
Arm = dict[str, Any]


def _make_batches(seed, steps):
    """Learnable synthetic data: each sample repeats a random period-8 base."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(steps):
        base = torch.randint(1, 256, (BATCH, PERIOD), generator=generator)
        batches.append(base.repeat(1, (SEQ + PERIOD - 1) // PERIOD)[:, :SEQ])
    return batches


def _freeze_1d(model):
    """Muon is a 2D optimizer; freeze every non-2D parameter first."""
    for param in model.parameters():
        if param.ndim != 2:
            param.requires_grad_(False)


def _unique_2d(model):
    """Unique 2D parameters in stable order (tied embed/lm_head de-duplicated)."""
    seen, out = set(), []
    for name, param in model.named_parameters():
        if param.ndim != 2 or id(param) in seen:
            continue
        seen.add(id(param))
        out.append((name, param))
    return out


def _cat(states):
    return torch.cat([state.reshape(-1) for state in states])


def _state(arm: Arm):
    """fp64 per-parameter internal state of an arm.

    The mantissa arm's internal state is the reconstructed precise value; every
    other arm's internal state is the parameter itself.
    """
    if arm["mantissa"]:
        states = []
        for param in arm["params"]:
            mantissa = arm["opt"].state.get(param, {}).get("mantissa")
            if mantissa is None:
                mantissa = torch.zeros_like(param, dtype=torch.uint16)
            states.append(_reconstruct_precise(param, mantissa).double())
        return states
    return [param.detach().double() for param in arm["params"]]


def _loss(model, tokens):
    res = model(tokens, labels=tokens)
    loss = res.loss
    aux = getattr(res, "aux_loss", None)
    if aux is not None:
        loss = loss + aux
    return loss


def _muon(params, lr, use_mantissa=False):
    return BatchedMuon(
        params, lr=lr, weight_decay=0.0, momentum=0.95, nesterov=True,
        ns_steps=5, adjust_lr_fn="match_rms_adamw", use_mantissa=use_mantissa,
    )


def _arm_params(named, ref_2d, name):
    params = [named[pname] for pname, _ in ref_2d]
    if len({id(param) for param in params}) != len(params):
        raise RuntimeError(f"arm {name!r}: tied mapping produced duplicate tensors")
    return params


def _quantize_momentum_bf16(arm: Arm):
    """Round an arm's momentum buffer onto the bf16 grid, values only, in place.

    ``copy_`` keeps the buffer tensor object (and its fp32 dtype) persistent, so
    ``BatchedMuon``'s mantissa promotion guard does not replace it; only the
    stored values are restricted to the bf16 grid, modelling 2-byte momentum.
    """
    for param in arm["params"]:
        state = arm["opt"].state.get(param)
        if state and "momentum_buffer" in state:
            state["momentum_buffer"].copy_(state["momentum_buffer"].to(torch.bfloat16))


def _quantize_momentum_mom24(arm: Arm):
    """Truncate an arm's momentum buffer to 24 bits, in place.

    Reinterprets the fp32 buffer as int32 and masks off the low 8 mantissa bits
    (``0xFFFFFF00``): truncation toward zero of the significand, keeping sign +
    exponent + 15 explicit mantissa bits (16 significant bits). The mask is
    applied through a bit view that shares storage, so the buffer object stays
    persistent; this models 3-byte momentum.
    """
    for param in arm["params"]:
        state = arm["opt"].state.get(param)
        if state and "momentum_buffer" in state:
            state["momentum_buffer"].view(torch.int32).bitwise_and_(-256)


def _build_optimizer_arm(name, ref_model, ref_2d, lr) -> Arm:
    """Deepcopy of the fp32 ref, truncated to the arm dtype (no round-to-near)."""
    arm_model = copy.deepcopy(ref_model)
    named = dict(arm_model.named_parameters())
    if name == "fp32b":
        # Independent fp32 duplicate: no cast, no mantissa seeding.
        params = _arm_params(named, ref_2d, name)
        opt = _muon(params, lr, use_mantissa=False)
        return {"name": name, "params": params, "mantissa": False,
                "bfmom": False, "mom24": False, "opt": opt, "model": None}
    mantissa = name in ("bf16_mantissa", "bf16_mantissa_bf16mom",
                        "bf16_mantissa_mom24")
    bfmom = name == "bf16_mantissa_bf16mom"
    mom24 = name == "bf16_mantissa_mom24"
    lows = {}
    for pname, _ in ref_2d:
        param = named[pname]
        high, low = _split_precise(param.detach())
        if name == "fp16":
            param.data = high.to(torch.float16)
        else:
            param.data = high
            if mantissa:
                lows[pname] = low
    params = _arm_params(named, ref_2d, name)
    opt = _muon(params, lr, use_mantissa=mantissa)
    if mantissa:
        for pname, low in lows.items():
            opt.state[named[pname]]["mantissa"] = low.clone()
    return {"name": name, "params": params,
            "mantissa": mantissa, "bfmom": bfmom, "mom24": mom24,
            "opt": opt, "model": None}


def _build_train_arm(name, ref_model, ref_2d, lr) -> Arm:
    """Deepcopy cast with the production round-to-nearest ``.to(dtype)``."""
    arm_model = copy.deepcopy(ref_model)
    dtype = {"bf16": torch.bfloat16, "bf16_mantissa": torch.bfloat16,
             "bf16_mantissa_bf16mom": torch.bfloat16,
             "bf16_mantissa_mom24": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32, "fp32b": torch.float32}[name]
    arm_model.to(dtype)
    named = dict(arm_model.named_parameters())
    params = _arm_params(named, ref_2d, name)
    mantissa = name in ("bf16_mantissa", "bf16_mantissa_bf16mom",
                        "bf16_mantissa_mom24")
    bfmom = name == "bf16_mantissa_bf16mom"
    mom24 = name == "bf16_mantissa_mom24"
    opt = _muon(params, lr, use_mantissa=mantissa)
    return {"name": name, "params": params,
            "mantissa": mantissa, "bfmom": bfmom, "mom24": mom24,
            "opt": opt, "model": arm_model}


def _run(lr, mode, arm_names, seed, steps, device, batches) -> list[Any]:
    torch.manual_seed(seed)
    ref_model = InstinctForCausalLM(cast(InstinctConfig, make_tiny_config())).to(device).train()
    _freeze_1d(ref_model)
    ref_2d = _unique_2d(ref_model)
    ref_params = [param for _, param in ref_2d]
    ref_arm: Arm = {"name": "fp32", "params": ref_params, "mantissa": False,
                    "bfmom": False, "mom24": False, "opt": _muon(ref_params, lr),
                    "model": ref_model}

    arms: list[Arm] = []
    for name in arm_names:
        builder = _build_optimizer_arm if mode == "optimizer" else _build_train_arm
        arms.append(builder(name, ref_model, ref_2d, lr))

    ref_origin = _state(ref_arm)
    ref_prev = [t.clone() for t in ref_origin]
    for arm in arms:
        arm["origin"] = _state(arm)
        arm["prev"] = [t.clone() for t in arm["origin"]]
        arm["series"] = {"accum_delta": [], "traj_delta": [], "cos": [],
                         "step_ratio": [], "max_abs": []}
        arm["zero_elem_num"] = 0
        arm["zero_elem_den"] = 0
        arm["exact_steps"] = 0
        arm["bit_exact_steps"] = 0

    num_params = len(ref_2d)

    for t in range(1, steps + 1):
        tokens = batches[t - 1].to(device)
        if mode == "optimizer":
            ref_arm["opt"].zero_grad(set_to_none=True)
            _loss(ref_model, tokens).backward()
            canonicals = []
            for pname, param in ref_2d:
                grad = param.grad
                if grad is None:
                    raise RuntimeError(f"reference gradient missing for {pname!r}")
                canonical = grad.detach().to(torch.bfloat16).to(torch.float32)
                param.grad = canonical
                canonicals.append(canonical)
            for arm in arms:
                for param, canonical in zip(arm["params"], canonicals):
                    param.grad = canonical.to(param.dtype)
            ref_arm["opt"].step()
            for arm in arms:
                arm["opt"].step()
                if arm["bfmom"]:
                    _quantize_momentum_bf16(arm)
                elif arm["mom24"]:
                    _quantize_momentum_mom24(arm)
        else:
            ref_arm["opt"].zero_grad(set_to_none=True)
            _loss(ref_model, tokens).backward()
            ref_arm["opt"].step()
            for arm in arms:
                arm["opt"].zero_grad(set_to_none=True)
                _loss(arm["model"], tokens).backward()
                arm["opt"].step()
                if arm["bfmom"]:
                    _quantize_momentum_bf16(arm)
                elif arm["mom24"]:
                    _quantize_momentum_mom24(arm)

        ref_cur = _state(ref_arm)
        u_ref = _cat([x - y for x, y in zip(ref_cur, ref_origin)])
        n_ref = u_ref.norm().clamp(min=EPS)
        ref_step = _cat([x - y for x, y in zip(ref_cur, ref_prev)]).norm()

        for arm in arms:
            cur = _state(arm)
            u_arm = _cat([x - y for x, y in zip(cur, arm["origin"])])
            accum_delta = ((u_arm - u_ref).norm() / n_ref).item()
            traj_delta = (_cat([x - y for x, y in zip(cur, ref_cur)]).norm()
                          / n_ref).item()
            denom = u_arm.norm() * u_ref.norm()
            cos = (torch.dot(u_arm, u_ref) / denom).item() if denom > 0 else float("nan")
            step_arm = _cat([x - y for x, y in zip(cur, arm["prev"])]).norm()
            ratio = (step_arm / ref_step.clamp(min=EPS)).item()
            max_abs = max((x - y).abs().max().item() for x, y in zip(cur, ref_cur))

            for j in range(num_params):
                ref_changed = ref_cur[j] != ref_prev[j]
                arm_unchanged = cur[j] == arm["prev"][j]
                arm["zero_elem_num"] += int((ref_changed & arm_unchanged).sum().item())
                arm["zero_elem_den"] += int(ref_changed.sum().item())

            arm["series"]["accum_delta"].append(accum_delta)
            arm["series"]["traj_delta"].append(traj_delta)
            arm["series"]["cos"].append(cos)
            arm["series"]["step_ratio"].append(ratio)
            arm["series"]["max_abs"].append(max_abs)

            if arm["name"] == "fp32b":
                if all(torch.equal(cur[j], ref_cur[j]) for j in range(num_params)):
                    arm["bit_exact_steps"] += 1

            if mode == "optimizer" and arm["mantissa"]:
                exact = all(
                    torch.equal(
                        _reconstruct_precise(p, arm["opt"].state.get(p, {}).get("mantissa")),
                        rp.detach(),
                    )
                    for p, (_, rp) in zip(arm["params"], ref_2d)
                )
                if exact:
                    arm["exact_steps"] += 1

            arm["prev"] = [x.clone() for x in cur]
        ref_prev = [x.clone() for x in ref_cur]

    results = []
    for arm in arms:
        ratios = arm["series"]["step_ratio"]
        results.append({
            "name": arm["name"],
            "series": arm["series"],
            "zero_elem_frac": (arm["zero_elem_num"] / arm["zero_elem_den"]
                               if arm["zero_elem_den"] else 0.0),
            "bit_exact_steps": arm["bit_exact_steps"],
            "bit_exact_frac": (arm["bit_exact_steps"] / steps
                               if arm["name"] == "fp32b" else None),
            "internal_exact_frac": (arm["exact_steps"] / steps
                                    if mode == "optimizer" and arm["mantissa"] else None),
        })
    return results


def _fmt(value):
    return "-" if value is None else f"{value:.4f}"


def _print_table(lr, mode, steps, results):
    print(f"\n{'=' * 112}")
    print(f"lr={lr:g}   mode={mode}   steps={steps}")
    print(f"{'arm':<24}{'accum_delta':>13}{'traj_delta':>13}{'cos':>9}"
          f"{'step_med':>11}{'step_p05':>11}{'zero_elem':>11}{'max_abs':>11}"
          f"{'internal_exact':>16}")
    sample_ts = sorted({ts for ts in (1, 10, 50, 100, steps) if ts <= steps})
    for res in results:
        series = res["series"]
        ratios = series["step_ratio"]
        med = statistics.median(ratios)
        p05 = torch.tensor(ratios, dtype=torch.float64).quantile(0.05).item()
        print(f"{res['name']:<24}{series['accum_delta'][-1]:>13.4e}"
              f"{series['traj_delta'][-1]:>13.4e}{series['cos'][-1]:>9.4f}"
              f"{med:>11.5f}{p05:>11.5f}{res['zero_elem_frac']:>11.5f}"
              f"{series['max_abs'][-1]:>11.4e}"
              f"{_fmt(res['internal_exact_frac']):>16}")
    print(f"growth accum_delta @ t={sample_ts}")
    for res in results:
        samples = [res["series"]["accum_delta"][ts - 1] for ts in sample_ts]
        print(f"  {res['name']:<24}" + "".join(f"{value:>12.4e}" for value in samples))
    fp32b = next((res for res in results if res["name"] == "fp32b"), None)
    if fp32b is not None:
        label = ("fp32b duplicate (harness determinism)" if mode == "optimizer"
                 else "fp32b duplicate (fp32-vs-fp32 noise floor)")
        print(f"{label}: bit-exact steps {fp32b['bit_exact_steps']}/{steps}, "
              f"final accum_delta {fp32b['series']['accum_delta'][-1]:.4e}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["optimizer", "train"], default="optimizer")
    parser.add_argument("--lrs", nargs="+", type=float, default=[5e-4, 1e-4, 2e-5])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--arms", type=str,
                        default="fp32,fp32b,bf16,bf16_mantissa,"
                                "bf16_mantissa_bf16mom,bf16_mantissa_mom24,fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    arms = [a.strip().replace("+", "_") for a in args.arms.split(",") if a.strip()]
    if "fp32" not in arms:
        arms = ["fp32"] + arms
    arm_names = [a for a in arms if a != "fp32"]
    unknown = [a for a in arm_names if a not in ARM_CHOICES]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown} (choices: {ARM_CHOICES})")

    batches = _make_batches(args.seed + 1, args.steps)

    # Model summary + the informational one-time production cast offset.
    torch.manual_seed(args.seed)
    probe = InstinctForCausalLM(cast(InstinctConfig, make_tiny_config())).train()
    _freeze_1d(probe)
    probe_2d = _unique_2d(probe)
    theta = torch.cat([p.detach().reshape(-1) for _, p in probe_2d])
    theta_cast = torch.cat([p.detach().to(torch.bfloat16).to(torch.float32).reshape(-1)
                            for _, p in probe_2d])
    cast_offset = ((theta_cast.double() - theta.double()).norm()
                   / theta.double().norm()).item()
    shapes = [tuple(p.shape) for _, p in probe_2d]
    del probe
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device == "cuda" else ""))
    print(f"mode: {args.mode}   steps: {args.steps}   seed: {args.seed}   lrs: {args.lrs}")
    print(f"2D params: {len(probe_2d)}   elements: {theta.numel()}   shapes: {shapes}")
    print("production one-time cast offset ||theta0.bf16.fp32 - theta0||/||theta0|| ="
          f" {cast_offset:.3e}")

    dump: dict[str, Any] = {"mode": args.mode, "steps": args.steps, "seed": args.seed,
            "arms": arms, "model": {"params_2d": len(probe_2d),
                                    "elements": theta.numel(),
                                    "shapes": [list(s) for s in shapes]},
            "cast_offset": cast_offset, "results": {}}

    for lr in args.lrs:
        results = _run(lr, args.mode, arm_names, args.seed, args.steps, device, batches)
        _print_table(lr, args.mode, args.steps, results)
        dump["results"][f"{lr:g}"] = {
            res["name"]: {
                "series": res["series"],
                "summary": {
                    "accum_delta_final": res["series"]["accum_delta"][-1],
                    "traj_delta_final": res["series"]["traj_delta"][-1],
                    "cos_final": res["series"]["cos"][-1],
                    "step_ratio_median": statistics.median(res["series"]["step_ratio"]),
                    "step_ratio_p05": torch.tensor(
                        res["series"]["step_ratio"], dtype=torch.float64
                    ).quantile(0.05).item(),
                    "zero_elem_frac": res["zero_elem_frac"],
                    "bit_exact_frac": res["bit_exact_frac"],
                    "max_abs_final": res["series"]["max_abs"][-1],
                    "internal_exact_frac": res["internal_exact_frac"],
                },
            }
            for res in results
        }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(dump, handle, indent=2)
        print(f"\njson written to {args.json}")


if __name__ == "__main__":
    main()
