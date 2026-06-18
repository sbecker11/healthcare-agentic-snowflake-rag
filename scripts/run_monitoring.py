"""Production monitoring demo: simulate retrieval-score distributions shifting
over time and show PSI flagging drift as the input distribution moves away from
the reference window.

Run:  python scripts/run_monitoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from member_nav.mlops import population_stability_index


def main() -> None:
    rng = np.random.default_rng(7)
    # Reference window: retrieval top-scores observed at deployment time.
    reference = rng.normal(0.25, 0.05, 1000)

    print(f"{'week':>4}  {'mean_shift':>10}  {'psi':>8}  severity")
    print("-" * 40)
    for week, shift in enumerate([0.00, 0.01, 0.03, 0.06, 0.12, 0.25], start=1):
        current = rng.normal(0.25 + shift, 0.05, 1000)
        report = population_stability_index(reference.tolist(), current.tolist())
        flag = "  <-- ALERT: refresh index / retrain" if report.severity == "significant" else ""
        print(f"{week:>4}  {shift:>10.2f}  {report.psi:>8.3f}  {report.severity}{flag}")

    print("\nPSI bands: <0.10 stable | 0.10-0.25 moderate | >0.25 significant")


if __name__ == "__main__":
    main()
