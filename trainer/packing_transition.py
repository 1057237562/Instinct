"""Exact in-epoch migration between non-packed rows and packed blocks."""

import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DistributedSampler

from dataset.sequence_bucket import (
    BUCKET_CALIBRATION_BATCH_SIZE,
    BUCKET_CALIBRATION_MEMORY_GB,
    BUCKET_CALIBRATION_SEQ_LEN,
    bucket_token_budget,
    packing_preprocess_workers,
)
from trainer.trainer_utils import Logger, SkipBatchSampler


def packing_data_config(args) -> dict:
    """Return resume-critical dataset settings stored with checkpoints."""
    target = bool(getattr(args, 'sequence_packing', 0))
    packing_mode = str(getattr(args, 'sequence_packing_mode', 'fixed'))
    if packing_mode not in ('fixed', 'bucket'):
        raise ValueError("sequence_packing_mode must be 'fixed' or 'bucket'")
    packing_num_proc = packing_preprocess_workers(
        getattr(args, 'packing_num_proc', 0),
    )
    return {
        'sequence_packing': target,
        'active_sequence_packing': target,
        'target_sequence_packing': target,
        'packing_alignment_pending': False,
        'packing_batch_size': int(getattr(args, 'packing_batch_size', 1000)),
        'packing_num_proc': packing_num_proc,
        'sequence_packing_mode': packing_mode,
        'seq_bucket': int(getattr(args, 'seq_bucket', 2)),
        'bucket_batch_strategy': 'vram_time_cost_v5_large_first' if packing_mode == 'bucket' else 'fixed',
        'bucket_gpu_memory_gb': float(getattr(args, 'bucket_gpu_memory_gb', 16.0)),
        'bucket_max_seq_len': int(getattr(args, 'bucket_max_seq_len', 16384)),
        'bucket_large_threshold': int(getattr(args, 'bucket_large_threshold', 8192)),
        'batch_size': int(getattr(args, 'batch_size', 1)),
        'max_seq_len': int(getattr(args, 'max_seq_len', 0)),
        'data_path': os.path.normcase(os.path.abspath(getattr(args, 'data_path', ''))),
    }


def validate_packing_resume(args, ckp_data) -> bool:
    """Validate packing settings and report whether a transition is needed."""
    if not ckp_data:
        return False
    current = packing_data_config(args)
    saved = ckp_data.get('data_config') or {}
    saved_active = bool(saved.get('active_sequence_packing', saved.get('sequence_packing', False)))
    saved_target = bool(saved.get('target_sequence_packing', saved.get('sequence_packing', False)))
    pending = bool(saved.get('packing_alignment_pending', False))
    if pending and saved_target != current['target_sequence_packing']:
        raise ValueError(
            'Cannot change the sequence-packing target while an alignment is pending: '
            f"checkpoint target={int(saved_target)}, requested={int(current['target_sequence_packing'])}."
        )
    transition = pending or saved_active != current['target_sequence_packing']
    if transition:
        # Reconstructing an in-epoch migration requires the original raw cursor
        # and exactly the same packed side after another pause/resume.  A first
        # switch from raw -> packed may choose either packing strategy because
        # no packed rows from the checkpoint need reconstruction yet.
        keys = ['batch_size', 'data_path']
        if not saved_active:
            keys.append('max_seq_len')
        if pending or saved_active:
            keys.extend(['packing_batch_size', 'sequence_packing_mode'])
            saved_mode = saved.get('sequence_packing_mode', 'fixed')
            if saved_mode == 'bucket':
                keys.extend([
                    'seq_bucket', 'bucket_batch_strategy',
                    'bucket_gpu_memory_gb', 'bucket_max_seq_len',
                    'bucket_large_threshold', 'packing_num_proc',
                ])
            else:
                keys.append('max_seq_len')
        for key in keys:
            if key in saved and saved[key] != current[key]:
                raise ValueError(
                    f'Cannot resume a packing transition with changed {key}: '
                    f"checkpoint={saved.get(key)!r}, requested={current[key]!r}."
                )
        return True
    if not saved_active:
        return False
    current_mode = current['sequence_packing_mode']
    saved_mode = saved.get('sequence_packing_mode', 'fixed')
    if saved_mode != current_mode:
        raise ValueError(
            'Cannot resume packed training with changed sequence_packing_mode: '
            f"checkpoint={saved_mode!r}, requested={current_mode!r}."
        )
    keys = ['packing_batch_size', 'data_path']
    if current_mode == 'bucket':
        keys.extend([
            'seq_bucket', 'bucket_batch_strategy',
            'bucket_gpu_memory_gb', 'bucket_max_seq_len',
            'bucket_large_threshold', 'packing_num_proc',
        ])
    else:
        keys.append('max_seq_len')
    for key in keys:
        legacy_default = 1 if key == 'seq_bucket' else (
            32768 if key == 'bucket_max_seq_len' else (
                8192 if key == 'bucket_large_threshold' else None
            )
        )
        saved_value = saved.get(key, legacy_default)
        if saved_value != current[key]:
            raise ValueError(
                f'Cannot resume packed training with changed {key}: '
                f"checkpoint={saved_value!r}, requested={current[key]!r}."
            )
    return False


def build_dataset_with_cache_barrier(factory, *, packing: bool):
    """Let rank 0 build an Arrow packing cache before other DDP ranks load it."""
    if not packing or not dist.is_initialized():
        return factory()
    dataset = factory() if dist.get_rank() == 0 else None
    dist.barrier()
    return dataset if dataset is not None else factory()


class SequencePackingPlan:
    """Rebuild a checkpoint epoch and migrate its untouched suffix to packing."""

    def __init__(self, args, ckp_data, dataset_factory):
        self.args = args
        self.dataset_factory = dataset_factory
        self.target = bool(getattr(args, 'sequence_packing', 0))
        self.packing_mode = str(getattr(args, 'sequence_packing_mode', 'fixed'))
        saved = (ckp_data or {}).get('data_config') or {}
        self.source = bool(saved.get('active_sequence_packing', saved.get('sequence_packing', False)))
        self.transition = validate_packing_resume(args, ckp_data)
        self.start_epoch = int((ckp_data or {}).get('epoch', 0))
        self.intra_epoch = self.transition and not self.source and self.target
        self.origin_step = int(saved.get(
            'packing_transition_origin_step', (ckp_data or {}).get('step', 0)
        ))
        self._datasets = {}
        self._loader_workers_logged = False
        self._runtime_buckets = {}
        self._seen_runtime_buckets = set()
        self._large_phase_ends = {}
        self._released_large_phases = set()
        if self.intra_epoch:
            Logger(
                '[Packing Resume] intra-epoch alignment enabled: '
                f'epoch={self.start_epoch + 1}, origin_step={self.origin_step}. '
                'Unaligned rows remain non-packed; the untouched aligned suffix is packed.'
            )
        elif self.transition:
            Logger(
                '[Packing Resume] the current packed epoch cannot be expanded back into '
                'original rows; packing will be disabled at the next epoch boundary.'
            )

    def mode_for_epoch(self, epoch: int) -> bool:
        return self.source if self.transition and epoch == self.start_epoch else self.target

    def loader_num_workers(self) -> int:
        """Return a memory-safe DataLoader worker count for this packing plan."""
        requested = int(getattr(self.args, 'num_workers', 0))
        bucket_workers = int(getattr(self.args, 'bucket_loader_workers', -1))
        bucket_training = self.target and self.packing_mode == 'bucket'
        if bucket_training:
            if bucket_workers >= 0:
                resolved = bucket_workers
            elif os.name == 'nt':
                resolved = 0
            else:
                resolved = requested
        else:
            resolved = requested
        resolved = max(0, resolved)
        if bucket_training and not self._loader_workers_logged:
            Logger(
                '[Packing Memory] training DataLoader workers: '
                f'requested={requested}, bucket_workers={resolved}. '
                'Packing preprocessing workers are separate and exit before training.'
            )
            self._loader_workers_logged = True
        return resolved

    def _dataset(self, mode: bool, sample_indices=None):
        if sample_indices is not None:
            return build_dataset_with_cache_barrier(
                lambda: self.dataset_factory(mode, sample_indices), packing=mode,
            )
        key = ('full', mode)
        if key not in self._datasets:
            self._datasets[key] = build_dataset_with_cache_barrier(
                lambda: self.dataset_factory(mode, None), packing=mode,
            )
        return self._datasets[key]

    def dataset_for_epoch(self, epoch: int):
        mode = self.mode_for_epoch(epoch)
        return self._dataset(mode), mode

    @staticmethod
    def _global_epoch_indices(length: int, epoch: int):
        """Reproduce the exact sampler order used before migration."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        generator = torch.Generator()
        generator.manual_seed(epoch if world_size > 1 else 42 + epoch)
        indices = torch.randperm(length, generator=generator).tolist()
        if world_size > 1:
            per_rank = math.ceil(length / world_size)
            total_size = per_rank * world_size
            padding = total_size - len(indices)
            if padding:
                indices += (indices * math.ceil(padding / max(len(indices), 1)))[:padding]
        return indices, world_size, rank

    @staticmethod
    def _batches(indices, batch_size: int, *, offset: int = 0):
        return [
            [offset + index for index in indices[pos:pos + batch_size]]
            for pos in range(0, len(indices), batch_size)
        ]

    @staticmethod
    def _fixed_batches(dataset, batch_size: int, epoch: int, *, offset: int = 0):
        """Reproduce the original fixed-packing sampler exactly."""
        if dist.is_initialized():
            sampler = DistributedSampler(dataset)
            sampler.set_epoch(epoch)
            order = list(sampler)
        else:
            generator = torch.Generator().manual_seed(42 + int(epoch))
            order = torch.randperm(len(dataset), generator=generator).tolist()
        return SequencePackingPlan._batches(order, batch_size, offset=offset)

    def _bucket_batches(self, dataset, batch_size: int, epoch: int, *, offset: int = 0):
        """Build homogeneous batches with an approximately fixed token budget.

        The token budget is calibrated from the measured 16GB result of batch
        12 at 2048 tokens, then scaled linearly with the user-provided per-GPU
        VRAM. Flash/SDPA attention does not materialize a B*L^2 score matrix, so
        its dominant training activations are closer to B*L.
        """
        ranges = getattr(dataset, 'bucket_ranges', None)
        if not ranges:
            # Legacy packed datasets used one fixed block length and therefore
            # did not publish bucket metadata.
            ranges = [{'start': 0, 'end': len(dataset)}]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        generator = torch.Generator().manual_seed(42 + int(epoch))
        large_batches = []
        regular_batches = []
        gpu_memory_gb = float(getattr(self.args, 'bucket_gpu_memory_gb', 16.0))
        if not math.isfinite(gpu_memory_gb) or gpu_memory_gb <= 0:
            raise ValueError('bucket_gpu_memory_gb must be a positive finite number')
        token_budget = bucket_token_budget(gpu_memory_gb)
        large_threshold = int(getattr(self.args, 'bucket_large_threshold', 8192))
        if large_threshold < 1:
            raise ValueError('bucket_large_threshold must be at least 1')
        bucket_batch_sizes = []
        for bucket in ranges:
            start, end = int(bucket['start']), int(bucket['end'])
            count = end - start
            max_length = int(bucket['max_length'])
            scaled_batch_size = max(1, token_budget // max_length)
            bucket_batch_sizes.append(scaled_batch_size)
            order = (torch.randperm(count, generator=generator) + start).tolist()
            if world_size > 1:
                padding = (-len(order)) % world_size
                if padding:
                    order += (order * math.ceil(padding / max(len(order), 1)))[:padding]
                order = order[rank::world_size]
            bucket_batches = SequencePackingPlan._batches(
                order, scaled_batch_size, offset=offset,
            )
            if max_length > large_threshold:
                large_batches.extend(bucket_batches)
            else:
                regular_batches.extend(bucket_batches)

        Logger(
            f'[Packing Batch Plan] epoch={epoch + 1}, memory_model=B*L, '
            f'gpu_memory={gpu_memory_gb:g}GB, token_budget={token_budget}, '
            f'calibration={BUCKET_CALIBRATION_MEMORY_GB:g}GB:'
            f'{BUCKET_CALIBRATION_SEQ_LEN}x{BUCKET_CALIBRATION_BATCH_SIZE}'
        )
        for number, (bucket, scaled_batch_size) in enumerate(
                zip(ranges, bucket_batch_sizes), start=1):
            Logger(
                f'[Packing Batch] {number}/{len(ranges)}: '
                f"max_seq_len={int(bucket['max_length'])}, "
                f'batch_size={scaled_batch_size}'
            )

        self._runtime_buckets[epoch] = {
            int(bucket['max_length']): {
                'number': number,
                'count': len(ranges),
                'planned_batch_size': scaled_batch_size,
            }
            for number, (bucket, scaled_batch_size) in enumerate(
                zip(ranges, bucket_batch_sizes), start=1
            )
        }

        # Keep long-shape CUDA graphs in one bounded phase.  Each group remains
        # shuffled deterministically, while all >threshold batches precede the
        # regular buckets so their shape-specific GPU allocations can be
        # destroyed at one known step boundary.
        for group in (large_batches, regular_batches):
            if group:
                permutation = torch.randperm(len(group), generator=generator).tolist()
                group[:] = [group[index] for index in permutation]
        self._large_phase_ends[epoch] = len(large_batches)
        Logger(
            f'[Packing Large Phase] epoch={epoch + 1}, threshold={large_threshold}, '
            f'large_batches={len(large_batches)}, regular_batches={len(regular_batches)}, '
            'order=large-first'
        )
        batches = large_batches + regular_batches
        return batches

    def should_release_large_cuda_memory(self, *, epoch: int, step: int) -> bool:
        """Return true once, immediately after this epoch's long-bucket phase."""
        phase_end = self._large_phase_ends.get(epoch, 0)
        key = (epoch, phase_end)
        if phase_end <= 0 or step != phase_end or key in self._released_large_phases:
            return False
        self._released_large_phases.add(key)
        return True

    def has_large_phase(self, epoch: int) -> bool:
        return self._large_phase_ends.get(epoch, 0) > 0

    def observe_batch(self, batch, *, epoch: int, step: int) -> str:
        """Log the first real use of each bucket and describe this micro-batch."""
        buckets = self._runtime_buckets.get(epoch)
        if not buckets or not batch:
            return ''
        input_ids = batch[0]
        if not hasattr(input_ids, 'shape') or len(input_ids.shape) < 2:
            return ''
        max_length = int(input_ids.shape[-1])
        info = buckets.get(max_length)
        if info is None:
            # An intra-epoch raw -> packed transition can yield raw batches
            # before reaching the bucketed suffix.
            return ''
        actual_batch_size = int(input_ids.shape[0])
        key = (epoch, max_length)
        if key not in self._seen_runtime_buckets:
            Logger(
                f"[Packing Bucket Active] epoch={epoch + 1}, step={step}, "
                f"bucket={info['number']}/{info['count']}, "
                f"max_seq_len={max_length}, batch_size={actual_batch_size} "
                f"(planned={info['planned_batch_size']})"
            )
            self._seen_runtime_buckets.add(key)
        return (
            f"bucket: {info['number']}/{info['count']} "
            f"({max_length}x{actual_batch_size})"
        )

    def _packing_batches(self, dataset, batch_size: int, epoch: int, *, offset: int = 0):
        mode = getattr(dataset, 'packing_mode', None) or self.packing_mode
        if mode == 'fixed':
            return self._fixed_batches(dataset, batch_size, epoch, offset=offset)
        if mode == 'bucket':
            return self._bucket_batches(dataset, batch_size, epoch, offset=offset)
        raise ValueError(f'unknown sequence packing mode: {mode!r}')

    def batch_sampler(self, dataset, *, active_packing: bool, epoch: int,
                      batch_size: int, skip_batches: int = 0):
        """Return a resume-aware sampler, grouping packed rows by block length."""
        if active_packing:
            batches = self._packing_batches(dataset, batch_size, epoch)
            return batches[skip_batches:]
        if dist.is_initialized():
            sampler = DistributedSampler(dataset)
            sampler.set_epoch(epoch)
        else:
            generator = torch.Generator().manual_seed(42 + int(epoch))
            sampler = torch.randperm(len(dataset), generator=generator).tolist()
        return SkipBatchSampler(sampler, batch_size, skip_batches)

    def epoch_data(self, epoch: int, resume_step: int, batch_size: int):
        """Return the dataset and optional hybrid batch list for one epoch."""
        if not (self.intra_epoch and epoch == self.start_epoch):
            dataset, mode = self.dataset_for_epoch(epoch)
            return dataset, None, mode

        raw_dataset = self._dataset(False)
        global_indices, world_size, rank = self._global_epoch_indices(len(raw_dataset), epoch)
        consumed = self.origin_step * batch_size * world_size
        if consumed > len(global_indices):
            raise ValueError(
                f'Checkpoint step {self.origin_step} exceeds the non-packed epoch '
                f'length for batch_size={batch_size} and world_size={world_size}.'
            )

        # Each rank advances to the same packing-group boundary.  The prefix
        # consequently has equal row and batch counts on every DDP rank.
        alignment_unit = max(1, int(self.args.packing_batch_size)) * world_size
        aligned = min(
            math.ceil(consumed / alignment_unit) * alignment_unit,
            len(global_indices),
        )
        raw_local = global_indices[consumed:aligned][rank::world_size]
        raw_batches = self._batches(raw_local, batch_size)

        remaining_indices = global_indices[aligned:]
        datasets = [raw_dataset]
        packed_batches = []
        if remaining_indices:
            packed_dataset = self._dataset(True, remaining_indices)
            packed_offset = len(raw_dataset)
            datasets.append(packed_dataset)
            packed_batches = self._packing_batches(
                packed_dataset, batch_size, epoch, offset=packed_offset,
            )

        all_batches = raw_batches + packed_batches
        completed_after_origin = resume_step - self.origin_step
        if completed_after_origin < 0 or completed_after_origin > len(all_batches):
            raise ValueError(
                'Packing transition cursor is outside the reconstructed hybrid epoch: '
                f'origin={self.origin_step}, resume={resume_step}, '
                f'transition_batches={len(all_batches)}.'
            )
        remaining_batches = all_batches[completed_after_origin:]
        Logger(
            '[Packing Resume] aligned inside epoch: '
            f'raw_rows_per_rank={len(raw_local)}, raw_batches={len(raw_batches)}, '
            f'packed_raw_rows={len(remaining_indices)}, packed_batches={len(packed_batches)}, '
            f'resume_batch_offset={completed_after_origin}.'
        )
        dataset = ConcatDataset(datasets) if len(datasets) > 1 else raw_dataset
        return dataset, remaining_batches, False

    def update_checkpoint_config(self, data_config: dict, *, epoch: int, active: bool) -> None:
        pending = self.transition and epoch == self.start_epoch
        data_config.update({
            'sequence_packing': self.target,
            'active_sequence_packing': bool(active),
            'target_sequence_packing': self.target,
            'packing_alignment_pending': pending,
        })
        if pending and self.intra_epoch:
            data_config.update({
                'packing_transition_epoch': self.start_epoch,
                'packing_transition_origin_step': self.origin_step,
            })
        else:
            data_config.pop('packing_transition_epoch', None)
            data_config.pop('packing_transition_origin_step', None)
