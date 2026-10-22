"""Cost of the magnitude policy (manuscript Assumption 4.3) for the balanced mandate.

The stopped-wealth analysis applies |omega*_a^[i]| (Pi*/Pi - 1) in both regimes, while the MV-optimal
policy of Corollary 3.4 is short below target in stress. The assumption quotes a survival cost of at
most 0.6 points and a Laplace-versus-simulation gap below 0.4 points for the four mandates of Table 2;
this test checks both for the balanced mandate with the same calibration as the paper figures.
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

from goal_based_allocation import compute_survival
from goal_based_allocation.riccati_solver import simulate_mv_optimal


@pytest.fixture
def producer(monkeypatch):
    replication = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(replication))
    return importlib.import_module("generate_paper_figures")


@pytest.mark.slow
def test_magnitude_policy_cost_balanced_mandate(producer):
    cal = producer._calibrate_mandate(0.35, c=0.0, omega_0=1.0)
    ric, gap = cal['ric'], cal['gap']
    laplace = compute_survival(ric.T, gap.x0, gap)
    n_paths = 60_000
    magnitude = simulate_mv_optimal(ric, n_paths=n_paths, steps_per_year=260, seed=42, policy='magnitude')
    signed = simulate_mv_optimal(ric, n_paths=n_paths, steps_per_year=260, seed=42, policy='signed')
    s_mag, s_sig = magnitude['survived'].mean(), signed['survived'].mean()
    standard_error = np.sqrt(s_mag * (1.0 - s_mag) / n_paths)
    assert abs(laplace - s_mag) < 0.004                      # analytics describe the magnitude policy
    assert -0.002 < s_sig - s_mag < 0.010                     # the short in stress buys at most 1 point
    assert abs(s_sig - s_mag) < 0.006 + 3.0 * standard_error  # the manuscript quotes at most 0.6 points
    assert signed['Pi_T'].mean() - magnitude['Pi_T'].mean() < 0.6 + 0.3
    with pytest.raises(ValueError):
        simulate_mv_optimal(ric, n_paths=10, steps_per_year=1, seed=1, policy='long')
