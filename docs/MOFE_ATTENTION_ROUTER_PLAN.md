# Instinct MoFE + Attention Router research and implementation plan

## Scope and fixed assumptions

- This is a breaking architecture change. Backward compatibility with the old MoE checkpoint/conversion path is out of scope.
- There are `n` already-trained dense Instinct checkpoints; the default UI workflow assumes `n = 16`.
- All source checkpoints use the same tokenizer, layer count, hidden size, FFN width, and tensor naming.
- A shared/base checkpoint supplies embeddings, attention, normalization, and LM head weights.
- Every source checkpoint supplies one FFN expert at every Transformer layer.
- Expert FFNs are immutable during MoFE post-pretraining. Only the router, or the router plus shared backbone, is trainable.
- Work remains isolated on branch `codex/mofe-attention-router` in `D:\AI\Instinct-mofe-attention-router`; this branch must not be merged automatically.

## Research conclusions

### Mixture of Frozen Experts

Seo et al. define MoFE by transplanting FFNs from existing expert models into a sparse MoE and freezing those FFNs. Their experiments support a large reduction in trainable parameters and training time. The paper also reports an important negative result: post-pretraining frozen FFNs can reduce downstream medical performance, plausibly because shared layers change while frozen FFNs cannot co-adapt.

Therefore, post-pretraining is implemented here as an explicit experimental stage rather than assumed to be universally beneficial. The UI must expose both `router_only` and `router_shared`; evaluation must compare them against no post-pretraining.

Primary source: <https://aclanthology.org/2025.naacl-industry.28/>

### Attention Router

Yuan 2.0-M32 replaces a single linear gate with three projections into expert space. For token state `x`, it forms `Q = Wq x`, `K = Wk x`, and `V = Wv x`, computes an expert-by-expert affinity matrix, then maps `V` through that matrix to produce expert logits. Top-k selection is applied afterwards. Yuan uses 32 experts with top-2 activation.

This implementation uses the same token-local expert-space interaction with configurable temperature and top-k. For the default 16 experts, the router's `16 x 16` affinity is small relative to FFN computation.

Primary sources: <https://arxiv.org/abs/2405.17976> and <https://github.com/IEIT-Yuan/Yuan2.0-M32>

## Proposed architecture

For every Transformer layer:

1. The shared attention/residual stream produces token state `x`.
2. Attention Router produces `n` logits from `x` through expert-space Q/K/V attention.
3. Softmax, top-k, and optional selected-probability normalization produce routing weights.
4. Only selected frozen FFNs run for each token.
5. Outputs are weighted and accumulated back into the shared residual stream.

The objective is:

`L = L_LM + lambda_balance * L_balance + lambda_z * L_z`

`L_balance` discourages collapsed expert allocation; `L_z` regularizes router-logit magnitude. Routing utilization is logged per layer. Top-1 routing deliberately keeps the selected softmax probability instead of normalizing it to one, preserving a task-loss gradient to the router.

## Expert-bank contract

The training input is a JSON manifest:

```json
{
  "base_model": "D:/weights/base.pth",
  "experts": [
    {"name": "general", "domain": "general", "path": "D:/weights/general.pth"},
    {"name": "code", "domain": "code", "path": "D:/weights/code.pth"}
  ]
}
```

Relative paths resolve from the manifest directory. Loading fails early on missing files, incompatible tensor shapes, missing FFN projections, expert-count mismatch, or a changed manifest during resume.

## Checkpoint policy

- Frozen source weights remain external and are never duplicated into periodic training checkpoints.
- `out/*_mofe_delta.pth` contains config, manifest identity, and trainable model delta.
- `checkpoints/*_moe_resume.pth` additionally contains optimizer/scaler/epoch/step/data-packing state.
- Resume validates the manifest fingerprint before applying the delta.
- `scripts/materialize_mofe.py` reconstructs the expert bank plus delta and can emit a self-contained deployment state dict when portability is preferred over storage efficiency.

## WebUI workflow

Add `mofe_post_pretrain` as a first-class training type in `scripts/config_webui.py` with:

- shared base checkpoint picker;
- multiline frozen-expert editor with default 16 slots and optional names/domains;
- validated expert count, unique names, file existence, and topology preflight;
- router type, top-k, temperature, balance loss, z-loss, and training-scope controls;
- dataset, packing, precision, optimizer, resume, pause, logs, and launch support;
- generated manifest preview and saved run-local manifest;
- explicit warning that published MoFE evidence found post-pretraining can hurt and that a no-post-pretraining control is required.

## Implementation milestones

1. **Architecture core:** attention/linear routers, correct top-k gradients, balance/z losses, immutable FFN experts.
2. **Assembly and checkpoints:** manifest validation, dense-FFN transplantation, train-scope selection, compact delta/resume format.
3. **Trainer:** post-pretraining dataset/packing/DDP/precision/pause/resume/logging flow.
4. **WebUI:** expert-bank editor, preflight, command construction, status and checkpoint handling.
5. **Verification:** unit tests for routing gradients, frozen weights, exact transplantation, invalid manifests, compact checkpoints, and WebUI command generation; CPU smoke training on tiny synthetic checkpoints.
6. **Research artifact:** paper draft with method, hypotheses, experimental matrix, limitations, and placeholder result tables. No empirical claim will be filled without measurements.

## Experimental matrix for the paper

At minimum compare:

- dense/shared baseline;
- classic trainable MoE;
- MoFE without post-pretraining;
- MoFE + linear router, `router_only` post-pretraining;
- MoFE + attention router, `router_only` post-pretraining;
- MoFE + attention router, `router_shared` post-pretraining;
- top-1 vs top-2 and 4/8/16 experts where compute permits.

Report held-out perplexity, downstream accuracy, trainable/total/active parameters, tokens per second, peak VRAM, checkpoint size, expert utilization entropy, dropped/imbalanced routing, and run-to-run variance.

## Acceptance criteria

- All expert parameters have `requires_grad = False`, receive no gradients, and remain bit-identical after an optimizer step.
- Router parameters receive nonzero task gradients for both top-1 and top-2.
- Every expert/layer FFN tensor matches its declared source checkpoint after assembly.
- Resume rejects any changed expert bank.
- WebUI can configure and launch a 16-expert run without hand-editing JSON or shell commands.
- Tests and a one-step CPU smoke run pass.
- A paper draft is present, clearly separating measured results from planned experiments.
