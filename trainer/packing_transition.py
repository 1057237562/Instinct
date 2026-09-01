"""Exact in-epoch migration between non-packed rows and packed blocks."""

import math
import os

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DistributedSampler

from trainer.trainer_utils import Logger, SkipBatchSampler


def packing_data_config(args) -> dict:
    """Return resume-critical dataset settings stored with checkpoints."""
    target = bool(getattr(args, 'sequence_packing', 0))
    packing_mode = str(getattr(args, 'sequence_packing_mode', 'fixed'))
    if packing_mode not in ('fixed', 'bucket'):
        raise ValueError("sequence_packing_mode must be 'fixed' or 'bucket'")
    return {
        'sequence_packing': target,
        'active_sequence_packing': target,
        'target_sequence_packing': target,
        'packing_alignment_pending': False,
        'packing_batch_size': int(getattr(args, 'packing_batch_size', 1000)),
        'sequence_packing_mode': packing_mode,
        'seq_bucket': int(getattr(args, 'seq_bucket', 2)),
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
            keys.append('seq_bucket' if saved_mode == 'bucket' else 'max_seq_len')
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
    keys.append('seq_bucket' if current_mode == 'bucket' else 'max_seq_len')
    for key in keys:
        saved_value = saved.get(key, 1 if key == 'seq_bucket' else None)
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

    @staticmethod
    def _bucket_batches(dataset, batch_size: int, epoch: int, *, offset: int = 0):
        """Build homogeneous-length batches and shard every bucket for DDP."""
        ranges = getattr(dataset, 'bucket_ranges', None)
        if not ranges:
            # Legacy packed datasets used one fixed block length and therefore
            # did not publish bucket metadata.
            ranges = [{'start': 0, 'end': len(dataset)}]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        generator = torch.Generator().manual_seed(42 + int(epoch))
        batches = []
        for bucket in ranges:
            start, end = int(bucket['start']), int(bucket['end'])
            count = end - start
            order = (torch.randperm(count, generator=generator) + start).tolist()
            if world_size > 1:
                padding = (-len(order)) % world_size
                if padding:
                    order += (order * math.ceil(padding / max(len(order), 1)))[:padding]
                order = order[rank::world_size]
            batches.extend(SequencePackingPlan._batches(order, batch_size, offset=offset))

        # All ranks use the same permutation of their corresponding batches,
        # retaining stochastic bucket order without ever mixing tensor shapes.
        if batches:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in permutation]
        return batches

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
