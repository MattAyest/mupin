"""Language-profile loader for the Editing Module.

Copied from the coding module with minimal edits. Profiles own all
language-specific conventions: file layout, sandbox image, tooling commands,
and LLM system prompts. The core pipeline consumes values through a ``Profile``
instance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml


PROFILE_DIR = Path(__file__).with_suffix("").parent.parent / "profiles"
DEFAULT_PROFILE = "python"

_PLACEHOLDER_KEYS = {
    "source_main",
    "test_main",
    "source_init",
    "test_init",
}


class ProfileError(Exception):
    """Raised when a profile is missing required keys or cannot be loaded."""


class Profile:
    """Runtime view of a language profile."""

    def __init__(self, data: Dict[str, Any], requested_name: str | None = None):
        self._data = data
        self.requested_name = requested_name
        self._validate()

    @property
    def name(self) -> str:
        return self._data.get("name", self.requested_name or DEFAULT_PROFILE)

    @property
    def display_name(self) -> str:
        return self._data.get("display_name", self.name)

    @property
    def sandbox(self) -> Dict[str, Any]:
        return self._data.get("sandbox", {})

    @property
    def files(self) -> Dict[str, Any]:
        return self._data.get("files", {})

    @property
    def workspace(self) -> Dict[str, Any]:
        return self._data.get("workspace", {})

    @property
    def prompts(self) -> Dict[str, Any]:
        return self._data.get("prompts", {})

    def file_path(self, key: str) -> str:
        """Return a concrete file path such as ``src/main.py``."""
        return str(self.files.get(key, key))

    def sandbox_value(self, key: str, default: Any = None) -> Any:
        """Read a value from the ``sandbox`` section with a fallback."""
        return self.sandbox.get(key, default)

    def resolve(self, text: str) -> str:
        """Replace profile placeholders in an arbitrary string."""
        resolved = text
        for k in _PLACEHOLDER_KEYS:
            resolved = resolved.replace(f"__{k.upper()}__", self.file_path(k))

        resolved = resolved.replace("__IMPORT_PATH__", str(self.prompts.get("import_path", "")))
        resolved = resolved.replace("__DEPS_FILE__", str(self.prompts.get("deps_file", "")))
        resolved = resolved.replace("__SOURCE_FILE__", self.file_path("source_main"))
        resolved = resolved.replace("__TEST_FILE__", self.file_path("test_main"))
        return resolved

    def prompt(self, key: str) -> str:
        """Return a system prompt with placeholders resolved."""
        raw = self.prompts.get(key, "")
        if not isinstance(raw, str):
            raise ProfileError(f"Prompt '{key}' for profile '{self.name}' is not a string")
        return self.resolve(raw)

    def setup_files(self) -> Dict[str, str]:
        """Return the workspace setup files as filename->content."""
        return self.workspace.get("setup_files", {})

    def manifest_files(self) -> set[str]:
        """Return the set of files that must be present in the manifest."""
        return set(self.files.get("manifest", []))

    def _validate(self) -> None:
        required_sections = {"sandbox", "files", "prompts"}
        missing = required_sections - set(self._data.keys())
        if missing:
            raise ProfileError(f"Profile '{self.name}' missing sections: {sorted(missing)}")

        required_sandbox_keys = {"image", "install_command", "verify_command"}
        missing_sandbox = required_sandbox_keys - set(self.sandbox.keys())
        if missing_sandbox:
            raise ProfileError(
                f"Profile '{self.name}' missing sandbox keys: {sorted(missing_sandbox)}"
            )

        required_file_keys = {"source_main", "test_main", "manifest"}
        missing_files = required_file_keys - set(self.files.keys())
        if missing_files:
            raise ProfileError(
                f"Profile '{self.name}' missing files keys: {sorted(missing_files)}"
            )

        required_prompt_keys = {
            "analyze_system",
            "plan_system",
            "apply_system",
        }
        missing_prompts = required_prompt_keys - set(self.prompts.keys())
        if missing_prompts:
            raise ProfileError(
                f"Profile '{self.name}' missing prompts: {sorted(missing_prompts)}"
            )


def _load_profile(name: str) -> Profile:
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        raise ProfileError(f"Profile '{name}' not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ProfileError(f"Profile '{name}' is not a valid YAML mapping")
    return Profile(data, requested_name=name)


def get_profile(name: str | None = None) -> Profile:
    """Load a profile by name, falling back to the default if missing or invalid."""
    requested = name or DEFAULT_PROFILE
    try:
        return _load_profile(requested)
    except ProfileError as e:
        if requested == DEFAULT_PROFILE:
            raise
        print(f"[profile] {e}; falling back to '{DEFAULT_PROFILE}'", file=sys.stderr)
        return _load_profile(DEFAULT_PROFILE)
