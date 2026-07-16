"""Hierarchical configuration loader for the Planner module.

Same pattern as coding/editing modules: root .env first, module .env overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

MODULE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = MODULE_DIR.parent


def load_env_hierarchy() -> None:
    global_dotenv = ROOT_DIR / ".env"
    local_dotenv = MODULE_DIR / ".env"
    if global_dotenv.exists():
        load_dotenv(global_dotenv, override=False)
    if local_dotenv.exists():
        load_dotenv(local_dotenv, override=True)


def load_llm_config() -> Dict[str, Any]:
    local = MODULE_DIR / "llm_config.yaml"
    if local.exists():
        with open(local, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    root = ROOT_DIR / "llm_config.yaml"
    if root.exists():
        with open(root, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    root_example = ROOT_DIR / "llm_config.yaml.example"
    if root_example.exists():
        with open(root_example, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    raise FileNotFoundError(f"No llm_config.yaml found in {MODULE_DIR} or {ROOT_DIR}")


def load_module_registry() -> Dict[str, Any]:
    reg = MODULE_DIR / "module_registry.yaml"
    if reg.exists():
        with open(reg, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {"modules": []}