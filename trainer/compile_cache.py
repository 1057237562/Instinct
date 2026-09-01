"""Configure persistent torch.compile caches before importing PyTorch.

Training entry points import this module before ``torch`` (and before model
modules which import torch).  Keeping this file torch-free is intentional: on
Windows the trainers must still import ``datasets`` before PyTorch to avoid the
known pyarrow DLL conflict.
"""

import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_TORCH_COMPILE_CACHE_DIR = os.path.join(REPO_ROOT, '.cache', 'torch_compile')

# Respect an explicit user/cluster cache location while making the repository
# cache persistent by default.  Inductor also places its Triton cache beneath
# TORCHINDUCTOR_CACHE_DIR when TRITON_CACHE_DIR is not explicitly configured.
TORCH_COMPILE_CACHE_DIR = os.path.abspath(os.environ.setdefault(
    'TORCHINDUCTOR_CACHE_DIR', DEFAULT_TORCH_COMPILE_CACHE_DIR,
))
os.environ.setdefault('TORCHINDUCTOR_FX_GRAPH_CACHE', '1')
os.environ.setdefault('TORCHINDUCTOR_AUTOGRAD_CACHE', '1')
os.makedirs(TORCH_COMPILE_CACHE_DIR, exist_ok=True)
