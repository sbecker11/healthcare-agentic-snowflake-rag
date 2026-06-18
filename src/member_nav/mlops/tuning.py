"""Hyperparameter optimization (the outer search loop).

This is the concrete answer to "does training search over hyperparameters?":
the inner loop is the graph executing with a *fixed* config; this outer loop
searches the config space. Each Optuna trial:

    1. samples a RagConfig from SEARCH_SPACE,
    2. runs the full graph over the eval set (the inner loop, many times),
    3. logs params + metrics to the experiment tracker,
    4. returns the composite metric for the sampler to optimize.

Optuna uses TPE (a Bayesian / tree-structured Parzen estimator) by default, so
the search is smarter than grid or random: it models the validation surface and
proposes promising configs. After the sweep, the best config is registered and
run through the promotion gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import optuna

from ..config import RagConfig, SEARCH_SPACE
from ..knowledge import Retriever
from ..llm import LLM
from .evaluation import EvalResult, evaluate, passes_gate
from .registry import ConfigRegistry
from .tracking import Tracker

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class SweepResult:
    best_config: RagConfig
    best_metrics: EvalResult
    n_trials: int
    promoted: bool
    gate_reason: str
    registered_version: int | None


def _suggest(trial: optuna.Trial, name: str, spec: tuple):
    kind = spec[0]
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"unknown suggest kind {kind!r}")


def run_sweep(
    retriever: Retriever,
    eval_set: list[dict],
    n_trials: int = 20,
    llm: LLM | None = None,
    tracker: Tracker | None = None,
    registry: ConfigRegistry | None = None,
    seed: int = 42,
) -> SweepResult:
    llm = llm or LLM()
    tracker = tracker or Tracker()
    registry = registry or ConfigRegistry()

    best = {"composite": -1.0, "config": None, "metrics": None}

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: _suggest(trial, name, spec) for name, spec in SEARCH_SPACE.items()
        }
        config = RagConfig.from_dict(params)
        with tracker.run(run_name=f"trial-{trial.number}", params=params) as rec:
            result = evaluate(config, retriever, eval_set, llm)
            rec.log_metrics(result.to_dict())
        if result.composite > best["composite"]:
            best.update(composite=result.composite, config=config, metrics=result)
        return result.composite

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_config: RagConfig = best["config"]
    best_metrics: EvalResult = best["metrics"]

    # Registry + promotion gate.
    incumbent = registry.production()
    incumbent_metrics = None
    if incumbent is not None:
        incumbent_metrics = EvalResult(**incumbent.metrics)
    promote, reason = passes_gate(best_metrics, incumbent_metrics)
    rv = registry.register(
        best_config, best_metrics.to_dict(),
        note=f"best of {n_trials}-trial TPE sweep",
    )
    if promote:
        registry.promote(rv.version)

    return SweepResult(
        best_config=best_config,
        best_metrics=best_metrics,
        n_trials=n_trials,
        promoted=promote,
        gate_reason=reason,
        registered_version=rv.version,
    )
