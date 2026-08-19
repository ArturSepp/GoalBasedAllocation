"""
Variance swaps under the two-state regime-switching jump-diffusion.

Companion to `vanilla_option_pricer`. Where the vanilla pricer gives option
prices, this module gives the fair variance-swap strike in CLOSED FORM and, more
importantly, decomposes it into diffusion and jump-cumulant contributions. That
decomposition is the analytic probe of the JUMP-SIZE PREMIUM: the variance swap
turns the second risk-neutral jump moment into a single tradeable number.

MODEL (measure Q, mean convention eta = E[jump])
------------------------------------------------
Regime chi_t in {0 growth, 1 stress}, generator Lambda = [[-l01, l01],[l10, -l10]].
Between transitions dX = nu^i dt + sigma^i dW, X = log S. At transitions:
    0 -> 1 (crash)   : Delta X = -J^[0],  J^[0] ~ Exp(mean eta_0)
    1 -> 0 (recovery): Delta X = +J^[1],  J^[1] ~ Exp(mean eta_1),  eta_1 < 1.
Exponential jump moments: E[J^k] = k! eta^k, so E[J] = eta, E[J^2] = 2 eta^2.

FAIR STRIKE (quadratic-variation convention)
--------------------------------------------
A variance swap pays realized variance = (1/T) [X]_T, with
    [X]_T = int_0^T (sigma^chi)^2 dt  +  sum_{jumps} (Delta X)^2.
Taking Q-expectations and writing tau_i = E_Q[time in regime i] over [0,T],

    K_var(T) = (1/T) [ sigma_0^2 tau_0 + sigma_1^2 tau_1                  (diffusion)
                     + l01 * tau_0 * E[(J^[0])^2]                        (crash jumps)
                     + l10 * tau_1 * E[(J^[1])^2] ]                      (recovery jumps)
             = (1/T) [ sigma_0^2 tau_0 + sigma_1^2 tau_1
                     + 2 l01 eta_0^2 tau_0 + 2 l10 eta_1^2 tau_1 ].

The occupation times are closed-form for the 2-state chain. With
Ls = l01 + l10, pi_0 = l10/Ls, pi_1 = l01/Ls, g = (1 - e^{-Ls T})/Ls:
    start 0:  tau_0 = pi_0 T + pi_1 g,   tau_1 = pi_1 (T - g)
    start 1:  tau_0 = pi_0 (T - g),      tau_1 = pi_1 T + pi_0 g.

WHY THIS DETECTS THE SIZE PREMIUM
---------------------------------
The jump part of K_var is AFFINE in the compound second cumulants
{l01 eta_0^2, l10 eta_1^2}. A vanilla ATM quote pins the diffusion level and the
first (compensator) cumulant; one variance-swap quote adds a clean linear equation
in the second cumulant. The jump-size premium is the gap between the Q-value of
these cumulants and their physical (P) counterparts.

Honest limitation (the entanglement): the swap sees only the PRODUCTS
l * eta^2, never intensity and size separately. Splitting them needs the physical
intensity (P-vs-Q), exactly the identification issue behind the near-degeneracy of
the two-option demand. See `variance_risk_premium`.

LOG-CONTRACT CONVENTION AND THE SKEW PROBE
------------------------------------------
The replication (Demeterfi-Derman-Kamal-Zou / Carr-Wu) strike prices the
log contract, K_log = (2/T) E_Q[ int dS/S - d log S ]. It agrees with K_var only
for pure diffusion; each jump contributes 2 E[e^{DX}-1-DX] instead of E[DX^2], so

    K_log - K_var = (2/T) sum_i l_i tau_i E[ e^{DX} - 1 - DX - 1/2 DX^2 ]
                  ~ (1/T) sum_i l_i tau_i E[(DX)^3] / ... (third cumulant, crash skew).

`jump_skew_gap` returns this pure jump object — a second analytic probe, of the
THIRD cumulant, i.e. crash magnitude.
"""
from __future__ import annotations

# packages
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union

# project
try:
    from .vanilla_option_pricer import RiskNeutralParams, Regime
except ImportError:  # direct module execution: python variance_swap.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # prefer local sources
    from goal_based_allocation.vanilla_option_pricer import RiskNeutralParams, Regime


class VarianceConvention(str, Enum):
    QUADRATIC_VARIATION = 'quadratic_variation'   # what a variance swap pays
    LOG_CONTRACT = 'log_contract'                 # option-replicated strike


@dataclass(frozen=True)
class VarianceDecomposition:
    """Additive components of the fair variance (all annualised, variance units)."""
    diffusion: float          # (sigma_0^2 tau_0 + sigma_1^2 tau_1) / T
    jump_crash: float         # 2 l01 eta_0^2 tau_0 / T   (0->1 downward jumps)
    jump_recovery: float      # 2 l10 eta_1^2 tau_1 / T   (1->0 upward jumps)

    @property
    def jump_total(self) -> float:
        return self.jump_crash + self.jump_recovery

    @property
    def total(self) -> float:
        return self.diffusion + self.jump_crash + self.jump_recovery

    @property
    def fair_vol(self) -> float:
        return float(np.sqrt(self.total))

    @property
    def jump_fraction(self) -> float:
        """Share of fair variance coming from jumps (the size/intensity premium carrier)."""
        return self.jump_total / self.total


# ==============================================================================
# OCCUPATION TIMES
# ==============================================================================

def occupation_times(params: RiskNeutralParams,
                     ttm: float,
                     regime: Union[int, Regime] = Regime.GROWTH,
                     ) -> Tuple[float, float]:
    """Expected time (tau_0, tau_1) spent in each regime over [0, T].

    Closed form for the 2-state chain; tau_0 + tau_1 = T exactly, and
    tau_i / T -> stationary pi_i as T -> inf.
    """
    if ttm <= 0.0:
        raise ValueError(f"ttm must be positive, got {ttm!r}")
    regime = int(regime)
    if regime not in (0, 1):
        raise ValueError(f"regime must be 0 or 1, got {regime!r}")

    l01, l10 = params.lambda_01, params.lambda_10
    ls = l01 + l10
    pi0, pi1 = l10 / ls, l01 / ls
    g = (1.0 - np.exp(-ls * ttm)) / ls

    if regime == 0:
        tau0 = pi0 * ttm + pi1 * g
        tau1 = pi1 * (ttm - g)
    else:
        tau0 = pi0 * (ttm - g)
        tau1 = pi1 * ttm + pi0 * g
    return float(tau0), float(tau1)


# ==============================================================================
# FAIR STRIKE
# ==============================================================================

def decompose_variance(params: RiskNeutralParams,
                       ttm: float,
                       regime: Union[int, Regime] = Regime.GROWTH,
                       ) -> VarianceDecomposition:
    """Additive diffusion / crash-jump / recovery-jump variance components.

    Each jump term is 2 * intensity * eta^2 * (time in source regime) / T, i.e.
    affine in the compound second cumulant l * eta^2.
    """
    tau0, tau1 = occupation_times(params, ttm, regime)
    diffusion = (params.sigma_0 ** 2 * tau0 + params.sigma_1 ** 2 * tau1) / ttm
    jump_crash = 2.0 * params.lambda_01 * params.eta_0 ** 2 * tau0 / ttm
    jump_recovery = 2.0 * params.lambda_10 * params.eta_1 ** 2 * tau1 / ttm
    return VarianceDecomposition(diffusion=diffusion,
                                 jump_crash=jump_crash,
                                 jump_recovery=jump_recovery)


def variance_swap_strike(params: RiskNeutralParams,
                         ttm: float,
                         regime: Union[int, Regime] = Regime.GROWTH,
                         convention: Union[str, VarianceConvention]
                         = VarianceConvention.QUADRATIC_VARIATION,
                         as_vol: bool = False,
                         ) -> float:
    """Fair variance-swap strike (annualised).

    Parameters
    ----------
    params : RiskNeutralParams
    ttm : float
        Swap maturity in years.
    regime : int or Regime
        Current regime chi_0.
    convention : VarianceConvention
        QUADRATIC_VARIATION - what the swap pays, E_Q[[X]_T]/T.
        LOG_CONTRACT       - option-replicated strike (2/T) E_Q[int dS/S - dlogS].
    as_vol : bool
        If True, return sqrt(strike) (fair volatility) rather than variance.

    Returns
    -------
    float
        Fair variance (or volatility if as_vol).
    """
    convention = VarianceConvention(convention)

    if convention == VarianceConvention.QUADRATIC_VARIATION:
        strike = decompose_variance(params, ttm, regime).total
    else:
        tau0, tau1 = occupation_times(params, ttm, regime)
        nu0, nu1 = params.nu
        e0, e1 = params.eta_0, params.eta_1
        # E_Q[X_T - X_0] = diffusion drift + jump drift
        jump_drift = -params.lambda_01 * e0 * tau0 + params.lambda_10 * e1 * tau1
        e_log = nu0 * tau0 + nu1 * tau1 + jump_drift
        strike = 2.0 * (params.rate * ttm - e_log) / ttm

    return float(np.sqrt(strike)) if as_vol else float(strike)


def jump_skew_gap(params: RiskNeutralParams,
                  ttm: float,
                  regime: Union[int, Regime] = Regime.GROWTH,
                  ) -> float:
    """K_log - K_var: a pure jump object dominated by the third cumulant.

    Negative for a left-skewed (crash-dominated) jump structure; a direct analytic
    probe of crash magnitude, complementary to the second-cumulant probe in
    `decompose_variance`.
    """
    k_log = variance_swap_strike(params, ttm, regime, VarianceConvention.LOG_CONTRACT)
    k_var = variance_swap_strike(params, ttm, regime, VarianceConvention.QUADRATIC_VARIATION)
    return k_log - k_var


# ==============================================================================
# VARIANCE RISK PREMIUM  (P vs Q)
# ==============================================================================

@dataclass(frozen=True)
class VarianceRiskPremium:
    """Decomposition of the variance risk premium K_var^Q - K_var^P by source."""
    total: float              # K_var^Q - K_var^P
    diffusion: float          # from sigma differences (0 if sigmas shared)
    jump: float               # from l * eta^2 differences (intensity AND size)
    physical_var: float       # K_var^P
    risk_neutral_var: float   # K_var^Q

    @property
    def ratio(self) -> float:
        """Q/P variance ratio - the usual 'VRP' multiple."""
        return self.risk_neutral_var / self.physical_var


def variance_risk_premium(params_p: RiskNeutralParams,
                          params_q: RiskNeutralParams,
                          ttm: float,
                          regime: Union[int, Regime] = Regime.GROWTH,
                          ) -> VarianceRiskPremium:
    """Variance risk premium K_var^Q - K_var^P, split into diffusion and jump parts.

    The jump part carries the compound cumulant difference
    (l01^Q eta0_Q^2 - l01^P eta0_P^2) and its recovery analogue: this is where BOTH
    the intensity premium and the size premium live, entangled as products. Options
    and variance swaps alone see only the product; separating intensity from size
    needs the physical intensity, which is precisely why this compares to P.
    """
    dq = decompose_variance(params_q, ttm, regime)
    dp = decompose_variance(params_p, ttm, regime)
    return VarianceRiskPremium(
        total=dq.total - dp.total,
        diffusion=dq.diffusion - dp.diffusion,
        jump=dq.jump_total - dp.jump_total,
        physical_var=dp.total,
        risk_neutral_var=dq.total,
    )


# ==============================================================================
# SIZE-PREMIUM CALIBRATION  (no-intensity-premium normalisation)
# ==============================================================================

@dataclass(frozen=True)
class SizePremiumCalibration:
    """Result of the one-parameter crash-size calibration.

    Under the normalisation lambda_Q = lambda_P, eta_1^Q = eta_1^P (no intensity
    premium, no recovery-size premium), the single free risk-neutral parameter is
    the mean crash size eta_0^Q, identified in closed form by one variance-swap
    quote. The cost of the normalisation: if the market does carry an intensity
    premium, eta_0^Q ABSORBS it - prices and P&L are unaffected (they depend only
    on the compound cumulant lambda * eta^2), but the frequency/severity
    decomposition is then misattributed. `size_premium_ratio` should be read as a
    total-crash-premium in severity units.
    """
    eta_0_q: float            # implied risk-neutral mean crash size
    params_q: 'RiskNeutralParams'   # full Q-parameter set (only eta_0 differs from P)
    size_premium_ratio: float  # eta_0^Q / eta_0^P  (>= 1 under the crash-premium prior)
    jump_variance_share: float  # share of the market fair variance explained by jumps


def implied_crash_size_from_var_swap(params_p: RiskNeutralParams,
                                     market_var_strike: float,
                                     ttm: float,
                                     regime: Union[int, Regime] = Regime.GROWTH,
                                     enforce_crash_premium_sign: bool = True,
                                     ) -> SizePremiumCalibration:
    """Closed-form eta_0^Q from one variance-swap quote (QV convention).

    Inverts the affine strike identity

        T K_var = sigma_0^2 tau_0 + sigma_1^2 tau_1
                + 2 lambda_01 (eta_0^Q)^2 tau_0 + 2 lambda_10 eta_1^2 tau_1

    for eta_0^Q, holding every other parameter at its physical value:

        eta_0^Q = sqrt( (T K_var - diffusion - recovery-jump) / (2 lambda_01 tau_0) ).

    Parameters
    ----------
    params_p : RiskNeutralParams
        PHYSICAL parameters (P-estimated); rate is reused for the Q set.
    market_var_strike : float
        Market fair variance (annualised, variance units; e.g. 0.05659 = 23.79% vol).
        Quadratic-variation convention.
    ttm : float
        Swap maturity in years.
    regime : int or Regime
        Current regime chi_0.
    enforce_crash_premium_sign : bool
        If True, raise when the implied eta_0^Q < eta_0^P (negative size premium);
        if False, return it and let the caller decide.

    Returns
    -------
    SizePremiumCalibration

    Raises
    ------
    ValueError
        If the market strike is below the model's jump-free floor (diffusion +
        recovery-jump variance), so no non-negative crash size can match it; or if
        `enforce_crash_premium_sign` and the implied premium is negative; or if
        the implied eta_0^Q >= 1 in log space is fine but we still sanity-cap
        against eta_0^Q producing an invalid parameter set.
    """
    if market_var_strike <= 0.0:
        raise ValueError(f"market_var_strike must be positive, got {market_var_strike!r}")
    tau0, tau1 = occupation_times(params_p, ttm, regime)

    diffusion = params_p.sigma_0 ** 2 * tau0 + params_p.sigma_1 ** 2 * tau1
    recovery = 2.0 * params_p.lambda_10 * params_p.eta_1 ** 2 * tau1
    numerator = ttm * market_var_strike - diffusion - recovery

    if numerator <= 0.0:
        floor = (diffusion + recovery) / ttm
        raise ValueError(
            f"market variance {market_var_strike!r} is below the model's jump-free "
            f"floor {floor:.6f} (diffusion + recovery jumps): no non-negative crash "
            f"size can match it. Check the P-side estimates before proceeding.")

    eta_0_q = float(np.sqrt(numerator / (2.0 * params_p.lambda_01 * tau0)))

    if enforce_crash_premium_sign and eta_0_q < params_p.eta_0:
        raise ValueError(
            f"implied eta_0^Q = {eta_0_q:.4f} < physical eta_0 = {params_p.eta_0:.4f}: "
            f"negative crash-size premium. Either the P-side crash size is "
            f"over-estimated or the market is pricing less crash risk than the "
            f"physical model; pass enforce_crash_premium_sign=False to accept.")

    params_q = RiskNeutralParams(sigma_0=params_p.sigma_0,
                                 sigma_1=params_p.sigma_1,
                                 lambda_01=params_p.lambda_01,   # lambda_Q = lambda_P
                                 lambda_10=params_p.lambda_10,
                                 eta_0=eta_0_q,                  # the one free parameter
                                 eta_1=params_p.eta_1,           # no recovery premium
                                 rate=params_p.rate)
    d = decompose_variance(params_q, ttm, regime)
    return SizePremiumCalibration(eta_0_q=eta_0_q,
                                  params_q=params_q,
                                  size_premium_ratio=eta_0_q / params_p.eta_0,
                                  jump_variance_share=d.jump_fraction)


def skew_overidentification_test(params_p: RiskNeutralParams,
                                 calibration: SizePremiumCalibration,
                                 spot: float,
                                 strikes: np.ndarray,
                                 ttm: float,
                                 regime: Union[int, Regime] = Regime.GROWTH,
                                 ) -> 'np.ndarray':
    """Predicted-minus-physical implied-vol skew: the over-identification check.

    The calibration uses ONE quote (the variance swap); the put skew across
    strikes is then a prediction with no remaining freedom. Compare the returned
    model skew against the market skew: a systematic miss (model too flat) is
    evidence that the no-intensity-premium normalisation is too tight and the
    escape hatch (free lambda_Q with shrinkage, or hyperexponential-2 severity)
    is needed.

    Returns an array of shape (len(strikes), 2): column 0 the implied vols under
    the calibrated Q, column 1 under the physical parameters priced risk-neutrally
    (the no-premium benchmark). The difference across strikes is the pure
    size-premium skew signature.
    """
    try:
        from .vanilla_option_pricer import implied_vol
    except ImportError:  # direct module execution
        from goal_based_allocation.vanilla_option_pricer import implied_vol
    iv_q = implied_vol(calibration.params_q, spot, strikes, ttm, regime,
                       option_type='put')
    iv_p = implied_vol(params_p, spot, strikes, ttm, regime, option_type='put')
    return np.column_stack([iv_q, iv_p])


# ==============================================================================
# MONTE CARLO REFERENCE (exact quadratic variation, no discretisation bias)
# ==============================================================================

def variance_swap_strike_mc(params: RiskNeutralParams,
                            ttm: float,
                            regime: Union[int, Regime] = Regime.GROWTH,
                            convention: Union[str, VarianceConvention]
                            = VarianceConvention.QUADRATIC_VARIATION,
                            n_paths: int = 200_000,
                            seed: int = 11,
                            ) -> Tuple[float, float]:
    """Monte Carlo fair variance by accumulating EXACT quadratic variation per path.

    Diffusion QV over a regime sojourn is exactly sigma_i^2 * (sojourn length); jump
    QV is (Delta X)^2. No time grid, so there is no discretisation bias - a clean
    check of the closed form. Returns (strike, standard_error).
    """
    convention = VarianceConvention(convention)
    regime = int(regime)
    rng = np.random.default_rng(seed)
    sig2 = np.array([params.sigma_0 ** 2, params.sigma_1 ** 2])
    lam = np.array([params.lambda_01, params.lambda_10])
    eta = np.array([params.eta_0, params.eta_1])

    chi = np.full(n_paths, regime, dtype=int)
    t = np.zeros(n_paths)
    acc = np.zeros(n_paths)                 # accumulated realized-variance integrand
    alive = np.ones(n_paths, dtype=bool)

    use_qv = convention == VarianceConvention.QUADRATIC_VARIATION

    while alive.any():
        idx = np.where(alive)[0]
        cur = chi[idx]
        dt_jump = rng.exponential(1.0 / lam[cur])
        dt_left = ttm - t[idx]
        dt = np.minimum(dt_jump, dt_left)
        acc[idx] += sig2[cur] * dt               # diffusion QV (exact)
        t[idx] += dt
        jumped = dt_jump < dt_left
        ji = idx[jumped]
        if len(ji) > 0:
            cj = chi[ji]
            size = rng.exponential(eta[cj])       # J >= 0
            dx = np.where(cj == 0, -size, size)   # crash down, recovery up
            if use_qv:
                acc[ji] += dx ** 2
            else:
                acc[ji] += 2.0 * (np.exp(dx) - 1.0 - dx)
            chi[ji] = 1 - cj
        alive[idx[~jumped]] = False

    rv = acc / ttm
    return float(rv.mean()), float(rv.std() / np.sqrt(n_paths))


# ==============================================================================
# DEMO, CHECKS AND TESTS
# ==============================================================================

class UnitTests(Enum):
    DEMO_STRIKE_DECOMPOSITION = 1
    DEMO_SIZE_PREMIUM = 2
    CHECK_OCCUPATION_TIMES = 3
    CHECK_NO_JUMP_LIMIT = 4
    TEST_MONTE_CARLO_QV = 5
    TEST_MONTE_CARLO_LOG = 6
    DEMO_IMPLIED_CRASH_SIZE = 7
    DEMO_SKEW_OVERIDENTIFICATION = 8


def paper_params() -> RiskNeutralParams:
    """Risk-neutral base case (Sturm's parameters, rate convention)."""
    return RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                        lambda_01=0.10, lambda_10=1.0,
                                        eta_01=3.0, eta_10=8.0, rate=0.03)


def run_local_test(unit_test: UnitTests) -> None:

    params = paper_params()
    ttm = 1.0

    if unit_test == UnitTests.DEMO_STRIKE_DECOMPOSITION:
        for regime in (Regime.GROWTH, Regime.STRESS):
            d = decompose_variance(params, ttm, regime)
            print(f"\n{regime.name} regime, T={ttm:g}:")
            print(f"  fair vol            {100 * d.fair_vol:6.2f}%   "
                  f"(fair variance {d.total:.5f})")
            print(f"  diffusion           {d.diffusion:.5f}   "
                  f"({100 * d.diffusion / d.total:5.1f}%)")
            print(f"  jump - crash        {d.jump_crash:.5f}   "
                  f"({100 * d.jump_crash / d.total:5.1f}%)")
            print(f"  jump - recovery     {d.jump_recovery:.5f}   "
                  f"({100 * d.jump_recovery / d.total:5.1f}%)")
            print(f"  jump fraction       {100 * d.jump_fraction:5.1f}%")
            gap = jump_skew_gap(params, ttm, regime)
            print(f"  K_log - K_var       {gap:+.5f}   (third-cumulant / skew probe)")

    elif unit_test == UnitTests.DEMO_SIZE_PREMIUM:
        # Physical vs risk-neutral: fatter/more frequent crashes under Q.
        p_phys = RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                              lambda_01=0.10, lambda_10=1.0,
                                              eta_01=4.0, eta_10=8.0, rate=0.03)
        p_rn = RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                            lambda_01=0.15, lambda_10=1.0,
                                            eta_01=2.5, eta_10=8.0, rate=0.03)
        for T in (1.0, 5.0):
            vrp = variance_risk_premium(p_phys, p_rn, T, Regime.GROWTH)
            print(f"\nT={T:g}:  fair vol  P={100 * np.sqrt(vrp.physical_var):.2f}%  "
                  f"Q={100 * np.sqrt(vrp.risk_neutral_var):.2f}%   ratio={vrp.ratio:.2f}")
            print(f"   VRP total     {vrp.total:+.5f}")
            print(f"   from diffusion {vrp.diffusion:+.5f}")
            print(f"   from jumps     {vrp.jump:+.5f}   "
                  f"(intensity+size, entangled as l*eta^2)")

    elif unit_test == UnitTests.CHECK_OCCUPATION_TIMES:
        for regime in (Regime.GROWTH, Regime.STRESS):
            for T in (0.5, 1.0, 5.0, 50.0):
                t0, t1 = occupation_times(params, T, regime)
                assert abs(t0 + t1 - T) < 1e-10, "occupation times must sum to T"
            # long-horizon -> stationary
            t0, t1 = occupation_times(params, 1e4, regime)
            ls = params.lambda_01 + params.lambda_10
            pi0 = params.lambda_10 / ls
            assert abs(t0 / 1e4 - pi0) < 1e-3, "must converge to stationary pi_0"
        print("occupation times: sum to T and converge to stationary pi. PASS")

    elif unit_test == UnitTests.CHECK_NO_JUMP_LIMIT:
        # vanishing jumps + equal vols -> K_var = sigma^2, K_log = sigma^2
        flat = RiskNeutralParams(sigma_0=0.2, sigma_1=0.2,
                                 lambda_01=0.05, lambda_10=0.05,
                                 eta_0=1e-4, eta_1=1e-4, rate=0.03)
        kv = variance_swap_strike(flat, 1.0, Regime.GROWTH,
                                  VarianceConvention.QUADRATIC_VARIATION)
        kl = variance_swap_strike(flat, 1.0, Regime.GROWTH,
                                  VarianceConvention.LOG_CONTRACT)
        assert abs(kv - 0.04) < 1e-6, f"QV strike should be sigma^2=0.04, got {kv}"
        assert abs(kl - 0.04) < 1e-6, f"log strike should be sigma^2=0.04, got {kl}"
        print(f"no-jump limit: K_var={kv:.6f}, K_log={kl:.6f}, both -> 0.04. PASS")

    elif unit_test == UnitTests.TEST_MONTE_CARLO_QV:
        print(f"{'regime':>7} {'T':>5} {'closed':>10} {'MC':>10} {'MC se':>9} {'z':>7}")
        for regime in (Regime.GROWTH, Regime.STRESS):
            for T in (0.5, 2.0, 10.0):
                cf = variance_swap_strike(params, T, regime,
                                          VarianceConvention.QUADRATIC_VARIATION)
                mc, se = variance_swap_strike_mc(params, T, regime,
                                                 VarianceConvention.QUADRATIC_VARIATION,
                                                 n_paths=300_000)
                z = (cf - mc) / se
                print(f"{regime.name:>7} {T:5.1f} {cf:10.5f} {mc:10.5f} {se:9.5f} {z:7.2f}")
                assert abs(z) < 4.0, "closed form disagrees with MC (QV)"
        print("PASS")

    elif unit_test == UnitTests.TEST_MONTE_CARLO_LOG:
        print(f"{'regime':>7} {'T':>5} {'closed':>10} {'MC':>10} {'MC se':>9} {'z':>7}")
        for regime in (Regime.GROWTH, Regime.STRESS):
            for T in (0.5, 2.0, 10.0):
                cf = variance_swap_strike(params, T, regime,
                                          VarianceConvention.LOG_CONTRACT)
                mc, se = variance_swap_strike_mc(params, T, regime,
                                                 VarianceConvention.LOG_CONTRACT,
                                                 n_paths=300_000)
                z = (cf - mc) / se
                print(f"{regime.name:>7} {T:5.1f} {cf:10.5f} {mc:10.5f} {se:9.5f} {z:7.2f}")
                assert abs(z) < 4.0, "closed form disagrees with MC (log contract)"
        print("PASS")


    elif unit_test == UnitTests.DEMO_IMPLIED_CRASH_SIZE:
        # Physical side: eta_0 = 1/4 (mean crash -25% in log). Market quotes a
        # var swap richer than the P-model: back out the implied Q crash size.
        p_phys = RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                              lambda_01=0.10, lambda_10=1.0,
                                              eta_01=4.0, eta_10=8.0, rate=0.03)
        k_p = variance_swap_strike(p_phys, ttm, Regime.GROWTH)
        print(f"physical fair vol: {100 * np.sqrt(k_p):.2f}%  (eta_0^P = {p_phys.eta_0:.4f})")
        for mkt_vol in (0.24, 0.26, 0.28):
            cal = implied_crash_size_from_var_swap(p_phys, mkt_vol ** 2, ttm,
                                                   Regime.GROWTH)
            print(f"market vol {100 * mkt_vol:.0f}%:  eta_0^Q = {cal.eta_0_q:.4f}  "
                  f"(x{cal.size_premium_ratio:.2f} physical), "
                  f"jump share {100 * cal.jump_variance_share:.1f}%")
        # guard demos
        try:
            implied_crash_size_from_var_swap(p_phys, 0.03 ** 2, ttm, Regime.GROWTH)
        except ValueError as err:
            print(f"below-floor guard: {str(err)[:80]}...")
        try:
            implied_crash_size_from_var_swap(p_phys, k_p * 0.98, ttm, Regime.GROWTH)
        except ValueError as err:
            print(f"sign guard: {str(err)[:80]}...")

    elif unit_test == UnitTests.DEMO_SKEW_OVERIDENTIFICATION:
        p_phys = RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                              lambda_01=0.10, lambda_10=1.0,
                                              eta_01=4.0, eta_10=8.0, rate=0.03)
        cal = implied_crash_size_from_var_swap(p_phys, 0.26 ** 2, ttm, Regime.GROWTH)
        spot = 100.0
        strikes = np.array([60.0, 70.0, 80.0, 90.0, 100.0, 105.0])
        ivs = skew_overidentification_test(p_phys, cal, spot, strikes, ttm,
                                           Regime.GROWTH)
        print(f"calibrated eta_0^Q = {cal.eta_0_q:.4f} from the var swap alone;")
        print("put skew below is a PREDICTION (no remaining freedom):\n")
        print(f"{'K':>6} {'iv Q (pred)':>12} {'iv P (bench)':>13} {'premium skew':>13}")
        for k, (q, pp) in zip(strikes, ivs):
            print(f"{k:6.0f} {100 * q:11.2f}% {100 * pp:12.2f}% {100 * (q - pp):+12.2f}%")
        print("\ncompare column 1 to the market skew: a systematic miss means the")
        print("no-intensity-premium normalisation is too tight (escape hatch: free")
        print("lambda_Q with shrinkage, or hyperexponential-2 severity).")


if __name__ == '__main__':

    run_local_test(unit_test=UnitTests.DEMO_STRIKE_DECOMPOSITION)