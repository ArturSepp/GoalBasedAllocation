"""
Goal-Based Allocation under Regime-Switching Jump-Diffusions.

Companion code to: "Dynamic Mean-Variance Portfolio Allocation under
Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution
Matching" (Sepp, 2026).

Modules
-------
laplace_inversion : Abate-Whitt and Stehfest numerical Laplace inversion.
regime_switch_paper : Core Laplace framework for transition densities.
riccati_solver : Riccati ODE system for MV-optimal allocation.
client_solver : Client profile specification and calibration.
mandate_utils : Portfolio mandate construction from individual assets.
opportunity_set : Investment opportunity set and two-step client framework.
variance_swap : Closed-form variance-swap strikes and jump-cumulant decomposition.
vanilla_option_pricer : Laplace-transform vanilla option pricing under regime switching.
"""

from .regime_switch_paper import (
    RegimeSwitchParams,
    AssetSpecification,
    MandateSpecification,
    compute_density,
    compute_survival,
    compute_tilted_survival,
    compute_overshoot_density,
    create_paper_assets,
    create_paper_mandates,
    bh_moments_rsjd,
)
from .riccati_solver import find_ell, gap_process_asset
from .client_solver import build_effective_asset, portfolio_sigma_unc, portfolio_eta_quadrature
from .opportunity_set import AdvisorSpec, compute_opportunity_point, build_opportunity_set
from .vanilla_option_pricer import (
    RiskNeutralParams,
    OptionType,
    Regime,
    price_vanilla,
    implied_vol,
)
from .variance_swap import (
    VarianceConvention,
    VarianceDecomposition,
    VarianceRiskPremium,
    SizePremiumCalibration,
    variance_swap_strike,
    decompose_variance,
    occupation_times,
    jump_skew_gap,
    variance_risk_premium,
    implied_crash_size_from_var_swap,
    skew_overidentification_test,
)

__version__ = "0.3.1"
