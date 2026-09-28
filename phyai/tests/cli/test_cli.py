"""Tests for the unified PhyAI CLI."""

from __future__ import annotations

import tomllib
from pathlib import Path

from phyai.cli import build_parser


def test_the_unified_cli_is_the_only_entry_point_and_exposes_doctor_and_env():
    assert build_parser().parse_args(["doctor"]).command == "doctor"
    env = build_parser().parse_args(["env", "--json"])
    assert env.command == "env" and env.json is True
    project = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["scripts"] == {"phyai": "phyai.cli:main"}
