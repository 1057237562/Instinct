"""Perplexity-guided LongRoPE evolutionary search for Instinct checkpoints.

This is a single-GPU implementation of Algorithm 1 from Ding et al., ICML
2024.  It searches one non-decreasing interpolation factor per rotary pair and
the paper's retained-start-token threshold (n-hat).  Run it only after a base
checkpoint exists: optimal factors depend on the learned model weights.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from transformers import AutoTokenizer

from model.rope import build_rope_caches, validate_rope_scaling


PAPER_RETAINED_TOKEN_CHOICES = (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128, 256)


@dataclass(frozen=True)
class Candidate:
    factors: tuple[float, ...]
    retained_start_tokens: int


def _round_to_step(values: np.ndarray, step: float) -> np.ndarray:
    return np.round(values / step) * step


def project_factors(values, scale: float, step: float) -> tuple[float, ...]:
    """Project a factor vector into the paper's monotonic search space."""
    maximum = scale * 1.25
    factors = np.asarray(values, dtype=np.float64)
    factors = np.clip(_round_to_step(factors, step), 1.0, maximum)
    factors = np.maximum.accumulate(factors)
    decimals = max(0, int(math.ceil(-math.log10(step)))) if step < 1 else 0
    factors = np.round(factors, decimals=decimals)
    return tuple(float(value) for value in factors)


def yarn_seed(dim: int, original_max: int, rope_theta: float, scale: float):
    """Construct the YaRN individual used in the paper's initial population."""
    inv_dim = lambda beta: (
        dim * math.log(original_max / (beta * 2 * math.pi))
    ) / (2 * math.log(rope_theta))
    low = max(math.floor(inv_dim(32.0)), 0)
    high = min(math.ceil(inv_dim(1.0)), dim // 2 - 1)
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float64) - low) / max(high - low, 0.001),
        0.0,
        1.0,
    )
    # YaRN scales inverse frequency by (1-ramp+ramp/scale), so lambda is
    # the reciprocal of that multiplier in LongRoPE notation.
    return 1.0 / (1.0 - ramp + ramp / scale)


def mutate_candidate(
    candidate: Candidate,
    rng: np.random.Generator,
    scale: float,
    step: float,
    probability: float,
) -> Candidate:
    factors = np.asarray(candidate.factors, dtype=np.float64).copy()
    mask = rng.random(factors.shape[0]) < probability
    if not mask.any():
        mask[rng.integers(0, factors.shape[0])] = True
    span = max(scale * 0.15, step)
    factors[mask] += rng.normal(0.0, span, int(mask.sum()))
    retained = candidate.retained_start_tokens
    if rng.random() < probability:
        retained = int(rng.choice(PAPER_RETAINED_TOKEN_CHOICES))
    return Candidate(project_factors(factors, scale, step), retained)


def crossover_candidates(
    left: Candidate,
    right: Candidate,
    rng: np.random.Generator,
    scale: float,
    step: float,
) -> Candidate:
    left_factors = np.asarray(left.factors)
    right_factors = np.asarray(right.factors)
    mask = rng.random(left_factors.shape[0]) < 0.5
    factors = np.where(mask, left_factors, right_factors)
    retained = left.retained_start_tokens if rng.random() < 0.5 else right.retained_start_tokens
    return Candidate(project_factors(factors, scale, step), retained)


def make_rope_scaling(candidate: Candidate, dim: int, original_max: int, target_length: int):
    scale = target_length / original_max
    return validate_rope_scaling(
        {
            "type": "longrope",
            "factor": scale,
            "original_max_position_embeddings": original_max,
            "short_factor": [1.0] * (dim // 2),
            "long_factor": list(candidate.factors),
            "short_attention_factor": 1.0,
            "short_retained_start_tokens": 0,
            "long_retained_start_tokens": candidate.retained_start_tokens,
        },
        dim,
        target_length,
    )


def _architecture_classes(architecture: str):
    if architecture == "linear":
        from model.model_instinct_linear import InstinctConfig, InstinctForCausalLM
    elif architecture == "looped":
        from model.model_instinct_loop import InstinctConfig, InstinctForCausalLM
    else:
        from model.model_instinct import InstinctConfig, InstinctForCausalLM
    return InstinctConfig, InstinctForCausalLM


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_validation_samples(path, tokenizer, target_length, count, rng):
    """Load complete long documents and crop deterministic target-length spans."""
    samples = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(samples) >= count:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str):
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) < target_length:
                continue
            start = int(rng.integers(0, len(token_ids) - target_length + 1))
            samples.append(torch.tensor(token_ids[start:start + target_length], dtype=torch.long))
    if len(samples) < count:
        raise ValueError(
            f"Only found {len(samples)} documents with at least {target_length} tokens in {path}; "
            f"the paper's search requires {count} long validation documents"
        )
    return samples


def install_candidate(model, scaling, target_length):
    backbone = model.model
    model.config.rope_scaling = scaling
    model.config.inference_rope_scaling = True
    model.config.max_position_embeddings = target_length
    backbone.config.rope_scaling = scaling
    backbone.config.inference_rope_scaling = True
    backbone.config.max_position_embeddings = target_length
    long_cos, long_sin, short_cos, short_sin = build_rope_caches(
        model.config.head_dim,
        target_length,
        model.config.rope_theta,
        scaling,
    )
    device = next(model.parameters()).device
    backbone.freqs_cos = long_cos.to(device)
    backbone.freqs_sin = long_sin.to(device)
    backbone.freqs_cos_short = short_cos.to(device)
    backbone.freqs_sin_short = short_sin.to(device)


@torch.inference_mode()
def evaluate_candidate(model, samples, scaling, target_length, device):
    install_candidate(model, scaling, target_length)
    losses = []
    for sample in samples:
        input_ids = sample.unsqueeze(0).to(device, non_blocking=True)
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        losses.append(float(output.loss.detach().cpu()))
        del input_ids, output
    return sum(losses) / len(losses)


def _unique(candidates):
    return list(dict.fromkeys(candidates))


def run_search(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with open(args.config_path, "r", encoding="utf-8") as stream:
        source_config = json.load(stream)
    architecture = source_config.get("model_architecture", "standard")
    config_class, model_class = _architecture_classes(architecture)
    original_max = int(args.original_length)
    target_length = int(args.target_length)
    if target_length <= original_max:
        raise ValueError("target_length must be greater than original_length")
    source_config["max_position_embeddings"] = target_length
    # Instantiate with native RoPE; candidate caches are installed after weights load.
    source_config["inference_rope_scaling"] = False
    source_config.pop("rope_scaling", None)
    source_config.pop("rope_parameters", None)
    model_config = config_class(**source_config)
    model = model_class(model_config)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_unwrap_state_dict(checkpoint), strict=True)

    device = torch.device(args.device)
    dtype = {
        "fp32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    samples = load_validation_samples(
        args.data_path, tokenizer, target_length, args.num_samples, rng
    )

    dim = model_config.head_dim
    scale = target_length / original_max
    maximum = scale * 1.25
    seeds = [
        Candidate(project_factors(np.ones(dim // 2), scale, args.step), 0),
        Candidate(project_factors(np.full(dim // 2, scale), scale, args.step), 0),
        Candidate(project_factors(yarn_seed(dim, original_max, model_config.rope_theta, scale), scale, args.step), 0),
    ]
    population = _unique(seeds)
    while len(population) < args.population_size:
        population.append(mutate_candidate(
            population[int(rng.integers(0, len(population)))],
            rng, scale, args.step, args.mutation_probability,
        ))
        population = _unique(population)

    scores = {}

    def score(candidate):
        if candidate not in scores:
            scaling = make_rope_scaling(candidate, dim, original_max, target_length)
            scores[candidate] = evaluate_candidate(
                model, samples, scaling, target_length, device
            )
            print(
                f"loss={scores[candidate]:.6f} n_hat={candidate.retained_start_tokens} "
                f"lambda=[{candidate.factors[0]:.2f}..{candidate.factors[-1]:.2f}]",
                flush=True,
            )
        return scores[candidate]

    for iteration in range(args.iterations):
        ranked = sorted(population, key=score)
        parents = ranked[:args.parents]
        best = parents[0]
        print(
            f"iteration={iteration + 1}/{args.iterations} best_loss={scores[best]:.6f} "
            f"n_hat={best.retained_start_tokens}",
            flush=True,
        )
        next_population = list(parents)
        for _ in range(args.mutations):
            parent = parents[int(rng.integers(0, len(parents)))]
            next_population.append(mutate_candidate(
                parent, rng, scale, args.step, args.mutation_probability
            ))
        for _ in range(args.crossovers):
            if len(parents) < 2:
                break
            left, right = rng.choice(len(parents), size=2, replace=False)
            next_population.append(crossover_candidates(
                parents[int(left)], parents[int(right)], rng, scale, args.step
            ))
        population = _unique(next_population)
        while len(population) < args.population_size:
            population.append(mutate_candidate(
                parents[int(rng.integers(0, len(parents)))],
                rng, scale, args.step, args.mutation_probability,
            ))
            population = _unique(population)

    best = min(population, key=score)
    best_scaling = make_rope_scaling(best, dim, original_max, target_length)
    output_config = copy.deepcopy(source_config)
    output_config["max_position_embeddings"] = target_length
    output_config["inference_rope_scaling"] = True
    output_config["rope_scaling"] = best_scaling
    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output_config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"best_loss={scores[best]:.6f}")
    print(f"saved={output_path}")
    print(f"search_range=[1.0, {maximum:.2f}], step={args.step}")


def build_parser():
    parser = argparse.ArgumentParser(description="Search model-specific LongRoPE factors")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--data_path", required=True, help="JSONL with one long document in each text field")
    parser.add_argument("--output_config", default="trainer/config_longrope.json")
    parser.add_argument("--original_length", type=int, default=4096)
    parser.add_argument("--target_length", type=int, default=32768)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--population_size", type=int, default=8)
    parser.add_argument("--parents", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--mutations", type=int, default=4)
    parser.add_argument("--crossovers", type=int, default=2)
    parser.add_argument("--mutation_probability", type=float, default=0.3)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "float16", "bfloat16"], default="bfloat16")
    return parser


if __name__ == "__main__":
    run_search(build_parser().parse_args())
