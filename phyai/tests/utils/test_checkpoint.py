"""Unit tests for phyai.utils.checkpoint folder helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from phyai.models.configuration import PretrainedConfig
from phyai.utils.checkpoint import (
    find_checkpoint_files,
    find_safetensors,
    load_config,
    resolve_checkpoint,
)


@dataclass(frozen=True)
class _TinyConfig(PretrainedConfig):
    hidden_size: int = 16
    name: str = "tiny"


def _index(tmp_path: Path, weight_map: dict) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


# --------------------------------------------------------------------------- #
# find_safetensors                                                            #
# --------------------------------------------------------------------------- #


def test_single_file_and_glob_fallback_return_absolute_paths(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"")  # empty placeholder
    out = find_safetensors(str(tmp_path))  # str is accepted too
    assert [p.name for p in out] == ["model.safetensors"] and out[0].is_absolute()

    other = tmp_path / "glob"
    other.mkdir()
    (other / "weights-a.safetensors").write_bytes(b"")
    (other / "weights-b.safetensors").write_bytes(b"")
    # No index, no canonical name: the glob picks up *.safetensors in order.
    assert [p.name for p in find_safetensors(other)] == [
        "weights-a.safetensors",
        "weights-b.safetensors",
    ]


def test_index_json_is_authoritative_and_deduplicates_shards(tmp_path: Path):
    save_file({"a": torch.zeros(2)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(
        {"b": torch.zeros(2), "c": torch.zeros(2)},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    (tmp_path / "model.safetensors").write_bytes(b"")  # the index wins over it
    _index(
        tmp_path,
        {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
            "c": "model-00002-of-00002.safetensors",
        },
    )
    out = find_safetensors(tmp_path)
    assert [p.name for p in out] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert all(p.is_absolute() for p in out)


@pytest.mark.parametrize(
    "index, error, message",
    [
        (
            {"weight_map": {"k": "ghost.safetensors"}},
            FileNotFoundError,
            "ghost.safetensors",
        ),
        ({"weight_map": {}}, ValueError, "weight_map"),
        ({"meta": {}}, ValueError, "weight_map"),
    ],
)
def test_a_broken_index_is_an_error(tmp_path: Path, index, error, message):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(error, match=message):
        find_safetensors(tmp_path)


def test_folders_without_shards_and_non_folders_are_errors(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no safetensors shards"):
        find_safetensors(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_safetensors(tmp_path / "nope")
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(NotADirectoryError, match="folder"):
        find_safetensors(tmp_path / "model.safetensors")


# --------------------------------------------------------------------------- #
# find_checkpoint_files                                                       #
# --------------------------------------------------------------------------- #


def test_checkpoint_files_prefer_canonical_safetensors_then_the_first_pytorch_format(
    tmp_path: Path,
):
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "extra.safetensors").write_bytes(b"")
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    assert [p.name for p in find_checkpoint_files(tmp_path)] == ["model.safetensors"]

    torch_only = tmp_path / "torch"
    torch_only.mkdir()
    for name in ("model.pth", "model.pt", "model.bin"):
        (torch_only / name).write_bytes(b"")
    assert [p.name for p in find_checkpoint_files(torch_only)] == ["model.bin"]

    sharded = tmp_path / "sharded"
    sharded.mkdir()
    (sharded / "weights-00002-of-00002.pth").write_bytes(b"")
    (sharded / "weights-00001-of-00002.pth").write_bytes(b"")
    out = find_checkpoint_files(sharded)
    assert [p.name for p in out] == [
        "weights-00001-of-00002.pth",
        "weights-00002-of-00002.pth",
    ]
    assert all(p.is_absolute() for p in out)


def test_checkpoint_files_ignore_training_state_and_honour_a_safetensors_index(
    tmp_path: Path,
):
    for name in ("training_args.bin", "optimizer.bin", "scheduler.pt", "model.pt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in find_checkpoint_files(tmp_path)] == ["model.pt"]

    indexed = tmp_path / "indexed"
    indexed.mkdir()
    _index(indexed, {"x": "missing.safetensors"})
    (indexed / "model.pth").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        find_checkpoint_files(indexed)

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "optimizer.pth").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="no supported model weight files"):
        find_checkpoint_files(empty)


# --------------------------------------------------------------------------- #
# load_config / resolve_checkpoint                                            #
# --------------------------------------------------------------------------- #


def test_load_config_reads_known_keys_and_drops_unknown_ones(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 32, "name": "x", "totally_unrelated": [1]})
    )
    cfg = load_config(tmp_path, _TinyConfig)
    assert (cfg.hidden_size, cfg.name) == (32, "x")
    (tmp_path / "geometry.json").write_text(json.dumps({"hidden_size": 7}))
    assert load_config(tmp_path, _TinyConfig, filename="geometry.json").hidden_size == 7
    assert (
        load_config(tmp_path, _TinyConfig, filename="geometry.json").name == "tiny"
    )  # default

    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config(tmp_path / "no_config", _TinyConfig) if (
            tmp_path / "no_config"
        ).mkdir() is None else None
    # A nonexistent path is resolved first: neither local nor a valid repo id.
    with pytest.raises(FileNotFoundError, match="not a valid HuggingFace repo id"):
        load_config(tmp_path / "ghost" / "sub", _TinyConfig)


def test_resolve_checkpoint_returns_local_paths_and_downloads_repo_ids(
    tmp_path: Path, monkeypatch
):
    assert resolve_checkpoint(tmp_path) == tmp_path
    assert resolve_checkpoint(str(tmp_path)) == tmp_path
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"")
    assert resolve_checkpoint(f) == f
    with pytest.raises(FileNotFoundError, match="not a valid HuggingFace repo id"):
        resolve_checkpoint(tmp_path / "ghost" / "sub")  # offline, no network

    seen: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    assert resolve_checkpoint("nvidia/Cosmos3-Nano", revision="abc123") == tmp_path
    assert (seen["repo_id"], seen["repo_type"], seen["revision"]) == (
        "nvidia/Cosmos3-Nano",
        "model",
        "abc123",
    )
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 64}))
    assert load_config("org/model", _TinyConfig, revision="r1").hidden_size == 64
    assert (seen["repo_id"], seen["revision"]) == ("org/model", "r1")
