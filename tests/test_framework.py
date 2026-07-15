"""Smoke tests for the core regime-switching analytical framework."""
import numpy as np

from goal_based_allocation import (
    create_paper_assets, compute_survival, bh_moments_rsjd, find_ell,
)


def test_survival_is_a_decreasing_probability():
    eq = create_paper_assets()['equity']
    surv = [compute_survival(t, eq.x0, eq) for t in (1.0, 2.0, 5.0, 10.0)]
    assert all(0.0 <= s <= 1.0 for s in surv)
    assert all(surv[i] >= surv[i + 1] for i in range(len(surv) - 1))


def test_riccati_initial_condition():
    eq = create_paper_assets()['equity']
    _, ric = find_ell(eq, 10.0, target_return=0.04, r=0.02, c=0.0)
    assert np.allclose(ric.a[:, 0], [1.0, 1.0], atol=1e-6)


def test_buy_and_hold_moments_positive():
    eq = create_paper_assets()['equity']
    bh = bh_moments_rsjd(10.0, 100.0, eq, c=0.0)
    assert bh['E'] > 0.0
    assert bh['Std'] > 0.0
