"""Conservative eligibility for the locally validated training CUDA kernels."""
from functools import lru_cache
import importlib.util
import torch


@lru_cache(maxsize=16)
def blackwell_training_kernels(device):
    device = torch.device(device)
    return (device.type == 'cuda' and torch.version.hip is None
            and torch.__version__.startswith('2.13.')
            and importlib.util.find_spec('triton') is not None
            and torch.cuda.get_device_capability(device)[0] == 12)
