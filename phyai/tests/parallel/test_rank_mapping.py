"""Group-local rank semantics in the torch.distributed backends."""

from __future__ import annotations

from unittest.mock import sentinel

import pytest

from phyai.parallel.backends import rank as rank_module


def test_group_rank_is_translated_to_global_and_validated(monkeypatch):
    monkeypatch.setattr(rank_module.dist, "get_world_size", lambda group: 3)
    monkeypatch.setattr(
        rank_module.dist,
        "get_global_rank",
        lambda group, rank: (11, 17, 23)[rank],
    )

    assert rank_module.group_rank_to_global(sentinel.process_group, 1) == 17
    with pytest.raises(ValueError, match="outside"):
        rank_module.group_rank_to_global(sentinel.process_group, 3)
    with pytest.raises(TypeError, match="must be an int"):
        rank_module.group_rank_to_global(sentinel.process_group, True)
