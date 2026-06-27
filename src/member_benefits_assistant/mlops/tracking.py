"""Experiment tracking.

A thin wrapper over MLflow so every evaluation / HPO trial logs its
hyperparameters and metrics to a queryable store (local ./mlruns file backend by
default). If MLflow is unavailable the wrapper degrades to appending JSON lines,
so tracking never blocks a run. This is the experiment-tracking + (lightweight)
registry layer of the MLOps lifecycle.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import mlflow

    _HAS_MLFLOW = True
except Exception:  # pragma: no cover
    _HAS_MLFLOW = False


class Tracker:
    def __init__(self, experiment: str = "member-benefits-assistant-rag",
                 tracking_uri: str | None = None,
                 fallback_path: str = "runs.jsonl"):
        self.experiment = experiment
        self.fallback_path = fallback_path
        self.backend = "mlflow" if _HAS_MLFLOW else "jsonl"
        if _HAS_MLFLOW:
            # MLflow 3.x retired the file store; SQLite is the portable default.
            default_uri = f"sqlite:///{Path('mlflow.db').resolve()}"
            uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", default_uri)
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, run_name: str, params: dict[str, Any]):
        if _HAS_MLFLOW:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                rec = _Recorder.mlflow()
                yield rec
        else:  # pragma: no cover
            rec = _Recorder.jsonl(self.fallback_path, run_name, params)
            yield rec
            rec.flush()


class _Recorder:
    def __init__(self, kind, **kw):
        self.kind = kind
        self._kw = kw
        self._metrics: dict[str, float] = {}

    @classmethod
    def mlflow(cls):
        return cls("mlflow")

    @classmethod
    def jsonl(cls, path, run_name, params):
        return cls("jsonl", path=path, run_name=run_name, params=params)

    def log_metrics(self, metrics: dict[str, float]):
        self._metrics.update(metrics)
        if self.kind == "mlflow":
            mlflow.log_metrics(metrics)

    def log_tag(self, key: str, value: str):
        if self.kind == "mlflow":
            mlflow.set_tag(key, value)

    def flush(self):  # pragma: no cover - jsonl path only
        if self.kind == "jsonl":
            rec = {
                "ts": time.time(),
                "run_name": self._kw["run_name"],
                "params": self._kw["params"],
                "metrics": self._metrics,
            }
            with open(self._kw["path"], "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
