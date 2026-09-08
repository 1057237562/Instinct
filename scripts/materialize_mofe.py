"""Materialize a compact MoFE delta into a self-contained deployment state dict."""

import argparse
import json
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.mofe_checkpoint import load_mofe_delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Materialize an Instinct MoFE delta")
    parser.add_argument("--delta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp16")
    args = parser.parse_args()

    model, payload = load_mofe_delta(args.delta, device="cpu")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    state = {
        name: tensor.detach().to(dtype=dtype, device="cpu") if tensor.is_floating_point() else tensor.cpu()
        for name, tensor in model.state_dict().items()
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(state, output)
    with open(output + ".config.json", "w", encoding="utf-8") as handle:
        json.dump(payload["config"], handle, ensure_ascii=False, indent=2)
    print(f"Materialized {len(state)} tensors to {output}")
