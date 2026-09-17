"""Shared bounded batches for decoder-only benchmark generation."""
import time


def generate_batches(args, model, tokenizer, jobs, batch_size, seed):
    """Yield (job, response, token_count); jobs contain text and a stable index."""
    from datasets import load_dataset  # noqa: F401; Windows DLL import order
    import torch
    from eval_llm import _generation_kwargs, setup_seed

    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    # Reuse tokenization across samples of the same question. All tensors here
    # remain on CPU until their batch is dispatched.
    tokenized = {}
    for job in jobs:
        if job['text'] not in tokenized:
            tokenized[job['text']] = tokenizer(job['text'], return_tensors='pt', truncation=False)
    ordered = sorted(jobs, key=lambda job: tokenized[job['text']]['input_ids'].shape[1]) if batch_size > 1 else jobs
    batches = [ordered[offset:offset + batch_size] for offset in range(0, len(ordered), batch_size)]
    if type(model).__module__ == 'model.model_instinct_linear' and batch_size > 1:
        # GatedDeltaNet's recurrent/convolution state does not consume a padding
        # mask. Never inject synthetic tokens into this backbone's context.
        groups = {}
        for job in jobs:
            length = tokenized[job['text']]['input_ids'].shape[1]
            groups.setdefault(length, []).append(job)
        batches = [group[offset:offset + batch_size] for group in groups.values()
                   for offset in range(0, len(group), batch_size)]
        print('[Eval] Linear backbone: batching equal-length prompts without padding.', flush=True)
    for batch_index, batch in enumerate(batches):
        started = time.perf_counter()
        encoded = [tokenized[job['text']] for job in batch]
        width = max(item['input_ids'].shape[1] for item in encoded)
        pad_id = getattr(tokenizer, 'pad_token_id', None)
        if pad_id is None:
            pad_id = getattr(tokenizer, 'eos_token_id', None)
        if pad_id is None and any(item['input_ids'].shape[1] != width for item in encoded):
            raise ValueError('Variable-length batches require a pad_token_id or eos_token_id')
        # Left padding keeps the last position on a real token for every row.
        ids, masks = [], []
        for item in encoded:
            tokens = item['input_ids']
            mask = item.get('attention_mask', torch.ones_like(tokens))
            padding = width - tokens.shape[1]
            ids.append(torch.nn.functional.pad(tokens, (padding, 0), value=pad_id or 0))
            masks.append(torch.nn.functional.pad(mask, (padding, 0), value=0))
        inputs = torch.cat(ids).to(args.device)
        attention_mask = torch.cat(masks).to(args.device)
        setup_seed(seed + batch[0]['index'])
        with torch.inference_mode():
            output = model.generate(inputs=inputs, attention_mask=attention_mask,
                                    **_generation_kwargs(args, tokenizer))
        if len(output) != len(batch):
            raise RuntimeError('Model returned a different number of rows than the input batch')
        # One device-to-host transfer per batch, instead of synchronizing CUDA
        # for each row's EOS search and tokenizer.decode.
        generated = output[:, width:].cpu()
        completed = []
        for job, tokens in zip(batch, generated):
            eos = getattr(tokenizer, 'eos_token_id', None)
            if eos is not None:
                positions = (tokens == eos).nonzero()
                if len(positions):
                    tokens = tokens[:int(positions[0].item()) + 1]
            completed.append((job, tokenizer.decode(tokens, skip_special_tokens=True), len(tokens)))
        elapsed = max(time.perf_counter() - started, 1e-9)
        pad_fraction = 1 - sum(item['input_ids'].shape[1] for item in encoded) / (len(batch) * width)
        print(f'[Eval batch {batch_index + 1}/{len(batches)}] sequences={len(batch)} '
              f'prompt_width={width} padding={pad_fraction:.1%} '
              f'generated_tokens={sum(row[2] for row in completed)} '
              f'tokens/s={sum(row[2] for row in completed) / elapsed:.1f} time={elapsed:.2f}s', flush=True)
        yield from completed
