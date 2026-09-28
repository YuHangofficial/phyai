"""Shared mesh and kernel-selector fixtures for phyai-kernel tests.

phyai-kernel's tests construct ``phyai.layers.*`` modules directly. Some of
those modules contain parallel linear layers, so they need a registered model
mesh. Kernel selection itself is lazy and needs no layer-specific setup.

We also need a registered :class:`Mesh` named ``"model"`` for layers
that resolve TP collectives (``ReplicatedLinear`` short-circuits at
ws=1 but still asks for the mesh by name). A degenerate single-rank
mesh covers every test in this package.

The autouse fixture below:

1. Registers a degenerate ``"model"`` mesh.
2. Isolates the process kernel selector.
3. Restores both process globals on exit.
"""

from __future__ import annotations

import pytest

from phyai.engine_config import ParallelConfig
from phyai.kernel.bootstrap import kernel_selector_scope
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh
from phyai.parallel.layout import build_rank_layout


def _register_fake_mesh(name: str = "model") -> Mesh:
    mesh = Mesh(build_rank_layout(ParallelConfig()), name=name)
    register_mesh(mesh)
    return mesh


@pytest.fixture(autouse=True)
def _phyai_layers_init():
    """Install a degenerate mesh and isolate kernel selection per test."""
    saved_meshes = dict(_meshes)
    _register_fake_mesh()
    try:
        with kernel_selector_scope():
            yield
    finally:
        _meshes.clear()
        _meshes.update(saved_meshes)
