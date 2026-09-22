from __future__ import annotations

import json
from pathlib import Path

from research_host.cli.agent_doctor import REQUIRED_AGENT_CONTEXT_KEYS


ROOT = Path(__file__).resolve().parents[1]


def test_agent_context_has_required_keys_and_start_files() -> None:
    context = json.loads((ROOT / "configs" / "agent_context.json").read_text(encoding="utf-8"))
    assert REQUIRED_AGENT_CONTEXT_KEYS.issubset(context)
    for relative in context["start_files"]:
        assert (ROOT / relative).exists(), relative
    for relative in context["public_contract_files"]:
        assert (ROOT / relative).exists(), relative
    for paths in context["task_routing"].values():
        for relative in paths:
            assert (ROOT / relative).exists(), relative


def test_agent_context_operation_classes_and_commands_are_structured() -> None:
    context = json.loads((ROOT / "configs" / "agent_context.json").read_text(encoding="utf-8"))
    assert context["schema_version"] == "1.0"
    assert context["operation_classes"]["safe_by_default"]
    assert context["operation_classes"]["requires_explicit_user_intent"]
    assert all(isinstance(command, str) for command in context["commands"].values())
    assert "Tracker feedback policy" in context["ask_before_changing"]


def test_local_agents_file_points_to_agent_guide() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "docs/AI_AGENT_GUIDE.md" in text
    assert "Tracker is ground truth only" in text
