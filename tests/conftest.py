"""Shared pytest fixtures and path setup.

The Fourier and Monte Carlo reference pricers live in ``examples/`` (they are
validation-only, not part of the shipped package). We add that folder to the
import path so the cross-check tests can import them.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))


@pytest.fixture(scope='session')
def rn_params():
    """Risk-neutral parameters from the paper's equity regime process (Table 1)."""
    from goal_based_allocation import create_paper_assets, RiskNeutralParams
    p = create_paper_assets()['equity'].params
    return RiskNeutralParams(sigma_0=p.sigma0,
                             sigma_1=p.sigma1,
                             lambda_01=p.lambda01,
                             lambda_10=p.lambda10,
                             eta_0=p.eta0,
                             eta_1=p.eta1,
                             rate=0.02)
