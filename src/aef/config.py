from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return resolve_paths(config)


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)
    base = Path(config["_config_dir"])

    def resolve(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else (base / path).resolve())

    data = config["data"]
    data["root"] = resolve(data["root"])
    data["stats_file"] = resolve(data["stats_file"])
    if "base_stats_file" in data:
        data["base_stats_file"] = resolve(data["base_stats_file"])
    if "temporal_data" in config:
        temporal = config["temporal_data"]
        temporal["temporal_root"] = resolve(temporal["temporal_root"])
        temporal["static_root"] = resolve(temporal["static_root"])
    for section in ("train", "downstream"):
        if section in config and "output_dir" in config[section]:
            config[section]["output_dir"] = resolve(config[section]["output_dir"])
    if "temporal_train" in config and "output_dir" in config["temporal_train"]:
        config["temporal_train"]["output_dir"] = resolve(
            config["temporal_train"]["output_dir"]
        )
    if "downstream" in config and "pretrained_checkpoint" in config["downstream"]:
        config["downstream"]["pretrained_checkpoint"] = resolve(
            config["downstream"]["pretrained_checkpoint"]
        )
    return config


def input_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: spec
        for name, spec in config["data"]["layers"].items()
        if spec["role"] in {"input", "input_target"}
    }


def target_specs(config: dict[str, Any], names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    result = {
        name: spec
        for name, spec in config["data"]["layers"].items()
        if spec["role"] in {"target", "input_target"}
    }
    if names is not None:
        result = {name: result[name] for name in names}
    return result
