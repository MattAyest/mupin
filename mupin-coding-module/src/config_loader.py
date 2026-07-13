"""Hierarchical configuration loader for the Coding Module.

Reads the global Mupin .env (repo root) first, then the module-local .env,
so that local values override global ones. The same pattern is used by the
Editing Module via its own copy of this helper.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


MODULE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = MODULE_DIR.parent


def load_env_hierarchy() -> None:
    """Load .env files in order: global first, then module-local overrides."""
    global_dotenv = ROOT_DIR / ".env"
    local_dotenv = MODULE_DIR / ".env"

    # Load global first so local overrides it.
    if global_dotenv.exists():
        load_dotenv(global_dotenv, override=False)
    if local_dotenv.exists():
        load_dotenv(local_dotenv, override=True)


def load_llm_config() -> Dict[str, Any]:
    """Load module-local llm_config.yaml; fall back to root example/default."""
    local_config = MODULE_DIR / "llm_config.yaml"
    if local_config.exists():
        with open(local_config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    root_config = ROOT_DIR / "llm_config.yaml"
    if root_config.exists():
        with open(root_config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    root_example = ROOT_DIR / "llm_config.yaml.example"
    if root_example.exists():
        with open(root_example, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise FileNotFoundError(
        f"No llm_config.yaml found in {MODULE_DIR} or {ROOT_DIR}"
    )
