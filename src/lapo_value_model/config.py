from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = str(path)
    for key in ("project_root", "data_root", "artifact_root", "output_root"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    for external in config.get("external_eval", {}).values():
        for key in ("data_root", "artifact_root"):
            if key in external:
                external[key] = str(Path(external[key]).expanduser().resolve())
    return config


def ensure_run_dirs(config: dict[str, Any]) -> None:
    for key in ("artifact_root", "output_root"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
