from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the frozen preparation contract is malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be a mapping")
    for section in ("contract", "source", "cleaning", "features", "splits", "augmentation"):
        if not isinstance(payload.get(section), dict):
            raise ConfigError(f"missing mapping section: {section}")
    payload["_config_path"] = str(config_path)
    payload["_workspace_root"] = str(config_path.parent.parent)
    return payload


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(public_config(config))).hexdigest()


def resolve_source_archive(config: dict[str, Any]) -> Path:
    configured = Path(str(config["source"]["archive"]))
    if configured.is_absolute():
        return configured.resolve()
    return (Path(config["_workspace_root"]) / configured).resolve()
