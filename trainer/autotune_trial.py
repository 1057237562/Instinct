"""Bounded, isolated measurements of the actual pretraining loop."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time
import traceback

from trainer.autotune_config import (
    SCHEMA, hardware_fingerprint, read_json, training_contract, validate_settings, write_json,
)


class TrialComplete(BaseException):
    pass


def prepare_trial(args):
    request = read_json(args.autotune_trial)
    if request.get('schema') != SCHEMA:
        raise ValueError('unknown autotune trial schema')
    validate_settings(request['settings'])
    if args.from_resume or args.use_wandb or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('autotune trials require single GPU, no resume and no external logging')
    if args.from_weight != 'none':
        raise ValueError('autotune currently calibrates isolated from-scratch pretraining')
    if not args.sequence_packing or args.sequence_packing_mode != 'bucket' or args.dataset_streaming != 'on':
        raise ValueError('autotune currently requires streaming bucket-packed pretraining')
    if args.use_grad_checkpoint not in (1, 2) or args.cache_build_mode != 'inline':
        raise ValueError('autotune keeps checkpointing on and uses the inline data pipeline')
    for name in ('warmup', 'steps', 'validation_examples'):
        if not isinstance(request[name], int) or request[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if args.streaming_prefetch_chunks and request['warmup'] < 2:
        raise ValueError('prefetch calibration needs at least two warmup steps')
    directory = Path(args.autotune_trial).resolve().parent
    if (Path(args.save_dir).resolve() != directory / 'weights'
            or Path(args.pause_file).resolve() != directory / 'pause'
            or args.save_interval <= request['warmup'] + request['steps']):
        raise ValueError('trial output/save isolation is not configured')
    source = Path(args.data_path).stat()
    if (source.st_size != request['source']['size'] or source.st_mtime_ns != request['source']['mtime_ns']):
        raise ValueError('corpus changed after audit')
    contract = training_contract(args)
    hardware = hardware_fingerprint(args.device)
    if request.get('hardware') and request['hardware'] != hardware:
        raise ValueError('hardware/software/source changed during search')
    if request.get('contract') and request['contract'] != contract:
        raise ValueError('a candidate changed the training contract')
    args._autotune_contract = contract
    if request.get('probe'):
        write_json(directory / 'result.json', dict(status='probe', hardware=hardware, contract=contract))
        raise SystemExit(0)


def validation_texts(source, plan, count):
    """Read a fixed held-out prefix of chunk 2, independent of packing budget."""
    if len(plan['chunks']) < 2:
        raise ValueError('autotune needs at least two source chunks for held-out validation')
    start = int(plan['chunks'][1]['start'])
    if plan['identity'].get('unit') == 'rows':
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(source)
        row = 0
        for group in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(group).num_rows
            if row + rows <= start:
                row += rows
                continue
            for batch in parquet.iter_batches(row_groups=[group], columns=['text'], batch_size=128):
                texts = batch.column(0).to_pylist()
                offset = max(0, start - row)
                for text in texts[offset:]:
                    yield str(text)
                    count -= 1
                    if count == 0:
                        return
                row += len(texts)
    else:
        import json
        with open(source, 'rb') as handle:
            handle.seek(start)
            for line in handle:
                if line.strip():
                    yield str(json.loads(line)['text'])
                    count -= 1
                    if count == 0:
                        return
    raise ValueError('not enough held-out validation records')


def run_pretrain_trial(namespace, request_path):
    import torch
    from trainer.streaming_pretrain import ChunkedPackedEpochLoader
    from trainer.windows_gpu_memory import local_memory_budget

    request = read_json(request_path)
    args, model, profiler = (namespace[key] for key in ('args', 'model', 'profiler'))
    raw = getattr(model, '_orig_mod', model)
    directory = Path(request_path).resolve().parent
    result = dict(schema=SCHEMA, status='error', contract=args._autotune_contract,
        settings=request['settings'], resolved_token_budget=args.bucket_token_budget,
        losses=[], observations=[], source=request['source'], weights_saved=False)
    optimizer = namespace['optimizer']
    muon = next((opt for opt in getattr(optimizer, 'optimizers', [optimizer])
                 if hasattr(opt, 'workspace_bytes')), None)
    defaults = dict(INSTINCT_MOE_INNER_CHECKPOINT='0', INSTINCT_MOE_SELECTIVE_CHECKPOINT='0',
        INSTINCT_MUON_PREALLOCATE='0', INSTINCT_MUON_WORKSPACE_MB=str(muon.workspace_bytes/2**20 if muon else 64),
        INSTINCT_MUON_BATCH_SIZE=str(muon.batch_size if muon else 16),
        INSTINCT_TRAINING_CPU_THREADS=str(torch.get_num_threads()),
        INSTINCT_GROUPED_MM_PROGRAMS_PER_SM='4', INSTINCT_GROUPED_MM_FWD_TILE='64,64,32,4,3',
        INSTINCT_GROUPED_MM_DW_TILE='64,64,32,4,3')
    result['resolved_settings'] = dict(args={k:getattr(args,k) for k in (
        'bucket_token_budget','bucket_gpu_memory_gb','use_grad_checkpoint','use_compile','compile_mode')},
        env={k:os.environ.get(k,v) for k,v in defaults.items()})
    old_end_step = profiler.end_step
    losses = []
    validation = []
    observed_batches = []
    def evaluate():
        raw.eval()
        total, tokens = 0., 0
        with torch.no_grad(), namespace['build_autocast_ctx'](args):
            for cpu_ids in validation:
                ids = cpu_ids.to(args.device)
                output = raw(ids, labels=ids)
                n = ids.numel() - 1
                total += float(output.loss) * n
                tokens += n
        raw.train()
        if not tokens or not math.isfinite(total):
            raise ValueError('non-finite validation loss or empty validation set')
        return total / tokens
    try:
        if namespace['lm_config'].model_architecture != 'standard':
            raise ValueError('autotune currently supports standard Dense/MoE pretraining')
        tokenizer = namespace['tokenizer']
        validation_hash = hashlib.sha256()
        for text in validation_texts(args.data_path, namespace['streaming_plan'], request['validation_examples']):
            ids = tokenizer(text, add_special_tokens=False, truncation=True,
                            max_length=args.bucket_max_seq_len - 2).input_ids
            tensor = torch.tensor([[tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]], dtype=torch.long)
            validation.append(tensor)
            validation_hash.update(tensor.numpy().tobytes())
        result['validation_sha256'] = validation_hash.hexdigest()
        weight_hash = hashlib.sha256()
        for name, parameter in raw.named_parameters():
            weight_hash.update(name.encode())
            weight_hash.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        result['initial_weights_sha256'] = weight_hash.hexdigest()
        result['validation_initial'] = evaluate()
        loader = ChunkedPackedEpochLoader(plan=namespace['streaming_plan'],
            dataset_factory=namespace['make_pretrain_dataset'], packing_plan=namespace['packing_plan'],
            args=args, epoch=0, data_config=namespace['data_config'])
        def end_step():
            # Called directly by the unchanged production train_epoch function.
            import sys
            # CUDA-graph output buffers can be reused on the next iteration.
            losses.append((sys._getframe(1).f_locals['loss'].detach() * args.accumulation_steps).clone())
            return old_end_step()
        profiler.end_step = end_step
        clocks = {}
        warm_shapes = set()
        def batches():
            iterator = iter(loader)
            try:
                for index in range(request['warmup'] + request['steps']):
                    if index == request['warmup']:
                        warm_cache_start = time.perf_counter()
                        loader.wait_for_pending_prefetch()
                        result['prefetch_warmup_seconds'] = time.perf_counter()-warm_cache_start
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        result['driver_budget_before'] = local_memory_budget(torch.cuda.get_device_name(args.device))
                        clocks['start'] = time.perf_counter()
                    batch = next(iterator)
                    # The trial never trains on its held-out source chunk.
                    if namespace['data_config'].get('streaming_chunk_index', 0) != 0:
                        raise ValueError('trial crossed into held-out chunk; reduce trial steps')
                    shape = tuple(batch[0].shape)
                    if index < request['warmup']:
                        warm_shapes.add(shape)
                    elif shape not in warm_shapes:
                        raise ValueError('new shape reached timed window; increase warmup')
                    observed_batches.append(batch)
                    yield batch
                    if (index+1) % 10 == 0:
                        print(f'[Autotune trial] completed={index+1}', flush=True)
                torch.cuda.synchronize()
                clocks['end'] = time.perf_counter()
                result.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
                result['driver_budget_after'] = local_memory_budget(torch.cuda.get_device_name(args.device))
                raise TrialComplete()
            except Exception:
                # Make the actual failure visible before production prefetch
                # cleanup joins a potentially long-running background pack.
                print(traceback.format_exc(), flush=True)
                raise
            finally:
                iterator.close()  # Join/finish production prefetch and release cache leases.
        try:
            namespace['train_epoch'](0, batches(), len(loader), 0, None, namespace['packing_plan'],
                token_schedule={'total_tokens': int(namespace['streaming_plan']['tokens'])*args.epochs})
        except TrialComplete:
            pass
        if 'end' not in clocks:
            raise RuntimeError('trial ended before the measurement boundary')
        result['losses'] = [float(x) for x in losses]
        if not all(math.isfinite(x) for x in result['losses']):
            raise ValueError('non-finite training loss')
        result['validation_final'] = evaluate()
        batch_hash = hashlib.sha256()
        useful, pairs = 0, 0
        for index, batch in enumerate(observed_batches):
            for tensor in batch:
                batch_hash.update(str(tuple(tensor.shape)).encode())
                batch_hash.update(tensor.numpy().tobytes())
            n = int((batch[1] != -100).sum())
            result['observations'].append(dict(shape=list(batch[0].shape), useful_tokens=n))
            if index >= request['warmup']:
                useful += n
                for row in batch[2]:
                    _, lengths = torch.unique_consecutive(row[row >= 0], return_counts=True)
                    pairs += int((lengths*(lengths+1)//2).sum())
        seconds = clocks['end'] - clocks['start']
        result.update(status='ok', batch_sha256=batch_hash.hexdigest(), wall_seconds=seconds,
            wall_ms_per_step=1000*seconds/request['steps'], useful_tokens=useful,
            useful_tokens_per_second=useful/seconds, warmup=request['warmup'], steps=request['steps'],
            peak_allocator_limit_gib=args.bucket_allocator_limit_gb,
            scope='real pretrain/Arrow/prefetch/LR/clip/optimizer; excludes initialization, warmup, validation and saves')
        peak = request.get('peak_tflops')
        if peak:
            config = namespace['lm_config']
            active = sum(p.numel() for p in raw.parameters())
            for module in raw.modules():
                experts = getattr(module, 'experts', None)
                if isinstance(experts, torch.nn.ModuleList):
                    active -= sum(p.numel() for p in experts.parameters()) * (1-config.num_experts_per_tok/len(experts))
            flops = 6*active*useful + 12*config.num_hidden_layers*config.num_attention_heads*config.head_dim*pairs
            result.update(mfu_estimate_pct=100*flops/seconds/(peak*1e12), reference_peak_tflops=peak)
        for info in (result['driver_budget_before'], result['driver_budget_after']):
            if info and info['current_usage'] > info['budget']:
                result['status'] = 'driver_budget_exceeded'
    except torch.cuda.OutOfMemoryError:
        result.update(status='oom', error=traceback.format_exc())
    except Exception:
        result.update(status='error', error=traceback.format_exc())
    finally:
        profiler.end_step = old_end_step
        write_json(directory / 'result.json', result)
    print(f'[AUTOTUNE RESULT] {result["status"]}; ' +
          str(result.get('useful_tokens_per_second', 'no timed result')), flush=True)
    return 0 if result['status'] == 'ok' else 2
