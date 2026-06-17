"""A minimal config/model registry with stage promotion.

Mirrors the MLflow Model Registry concept at the granularity that matters for
this system: a *config version* (the tuned hyperparameters + prompt version) is
the deployable artifact. Versions move through stages (None -> Staging ->
Production) and the gate in evaluation.py decides promotion. Persisted as JSON so
the registry is inspectable and diffable in git.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from ..config import RagConfig


@dataclass
class RegisteredVersion:
    version: int
    stage: str  # "None" | "Staging" | "Production" | "Archived"
    config: dict
    metrics: dict
    created_at: float = field(default_factory=time.time)
    note: str = ""


class ConfigRegistry:
    def __init__(self, path: str | Path = "registry.json"):
        self.path = Path(path)
        self._versions: list[RegisteredVersion] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._versions = [RegisteredVersion(**v) for v in raw]

    # -- queries -------------------------------------------------------------

    def production(self) -> RegisteredVersion | None:
        for v in reversed(self._versions):
            if v.stage == "Production":
                return v
        return None

    def latest(self) -> RegisteredVersion | None:
        return self._versions[-1] if self._versions else None

    def all(self) -> list[RegisteredVersion]:
        return list(self._versions)

    # -- mutations -----------------------------------------------------------

    def register(self, config: RagConfig, metrics: dict, note: str = "") -> RegisteredVersion:
        version = len(self._versions) + 1
        rv = RegisteredVersion(
            version=version, stage="Staging", config=config.to_dict(),
            metrics=metrics, note=note,
        )
        self._versions.append(rv)
        self._save()
        return rv

    def promote(self, version: int) -> RegisteredVersion:
        """Promote a version to Production, archiving the prior Production one."""
        target = self._get(version)
        for v in self._versions:
            if v.stage == "Production":
                v.stage = "Archived"
        target.stage = "Production"
        self._save()
        return target

    def _get(self, version: int) -> RegisteredVersion:
        for v in self._versions:
            if v.version == version:
                return v
        raise KeyError(f"version {version} not found")

    def _save(self):
        self.path.write_text(
            json.dumps([asdict(v) for v in self._versions], indent=2)
        )
