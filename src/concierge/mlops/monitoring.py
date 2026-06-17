"""Production monitoring: distribution drift detection.

The axis ordinary software has no analog for: a model's quality can decay with
zero code changes because the input distribution shifts underneath it. Here we
monitor the distribution of retrieval top-scores (a proxy for "are incoming
questions still answerable by the current knowledge base?") using the Population
Stability Index (PSI), the standard metric for tabular distribution shift.

PSI interpretation (industry convention):
    < 0.10  : no significant shift
    0.10-0.25 : moderate shift, investigate
    > 0.25  : significant shift, retrain / refresh index
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DriftReport:
    psi: float
    severity: str
    n_reference: int
    n_current: int

    def to_dict(self) -> dict:
        return {
            "psi": round(self.psi, 4),
            "severity": self.severity,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
        }


def _severity(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def population_stability_index(
    reference: list[float], current: list[float], bins: int = 10
) -> DriftReport:
    """Compute PSI between a reference window and a current window of scores.

    Uses equal-frequency (quantile) binning derived from the reference
    distribution, which is the standard, numerically stable PSI formulation and
    behaves well on modest window sizes. The effective bin count is reduced for
    small samples so bins are not starved of data.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return DriftReport(0.0, "stable", ref.size, cur.size)

    # Adaptive bin count: aim for >= ~5 reference points per bin.
    eff_bins = max(2, min(bins, ref.size // 5))

    # Equal-frequency edges from reference quantiles; open the tails to +/-inf.
    quantiles = np.linspace(0, 1, eff_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        return DriftReport(0.0, "stable", ref.size, cur.size)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, bins=edges)[0] / ref.size
    cur_pct = np.histogram(cur, bins=edges)[0] / cur.size

    # Laplace smoothing to avoid div-by-zero / log(0).
    eps = 1e-4
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return DriftReport(psi, _severity(psi), ref.size, cur.size)
