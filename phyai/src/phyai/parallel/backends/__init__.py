"""Backend implementations.

Currently registered:
  * :class:`NcclBackend`     — torch.distributed on NCCL PGs (eager + graph)
  * :class:`GlooBackend`     — torch.distributed on gloo PGs (eager only)
  * :class:`PyNCCLBackend`   — direct ctypes call to libnccl (graph-mode pref)

``can_handle`` of each backend gates on the PG's actual backend type so
they never fight over the same PG. Registration order in ``init`` decides
which one wins when multiple are eligible (e.g., NcclBackend vs PyNCCL on
the same NCCL PG; PyNCCL wins under capture).
"""

from __future__ import annotations

from phyai.parallel.backends.gloo import GlooBackend
from phyai.parallel.backends.pynccl import PyNCCLBackend
from phyai.parallel.backends.torch_dist import NcclBackend

__all__ = ["NcclBackend", "GlooBackend", "PyNCCLBackend"]
