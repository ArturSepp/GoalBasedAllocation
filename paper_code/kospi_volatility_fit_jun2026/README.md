# KOSPI 200 volatility fit — the "Korean frown" case (July 2026)

A calibration study of the stressed KOSPI 200 option surface, fitting the shipped
`GoalBasedAllocation` regime-switching jump-diffusion to real Bloomberg option chains
across three expiries, and testing whether the widely-circulated "Korean frown" is a
feature of the traded market or of Bloomberg's fitted BVOL surface.

**Verdict.** On the strikes that trade, the KOSPI 200 surface is a steep, elevated
**downside put skew** (with a mild call smile at the short end), not a frown. The
shipped exponential-jump model fits every tenor from the growth regime with a plain
crash jump — no Erlang or free-drift extension is needed, because there is no bimodality
to reproduce. The "frown" (ATM implied vol bid over both wings) is a property of the
smoothed BVOL **surface** — interpolation that sits outside the tradeable bid/ask, plus
extrapolation into deep-wing strikes the moneyness grid does not cover.

This directory is self-contained: raw chains, the transcribed BVOL grid, the fetcher,
the calibration and analysis code, and all figures.

---

## 1. Background

Two things motivated the study. First, a companion note (the regime-switch "frown"
framework) built its entire argument on **nine points digitised by eye from a Bloomberg
BVOL 3M screenshot**, and flagged its own gating prerequisite: *pull the real surface
and check whether a frown exists and whether its depth peaks near 3M*. Second, the same
3M BVOL curve was circulating publicly as "The Big Korean Frown" — ATM bid over both
wings — read as evidence of a bimodal, crash-or-recover risk-neutral density.

The market context is a genuine crisis, which the data confirms and which resolves the
"is ~70% vol real?" question: KOSPI 200 was down ~6.3% on the snapshot day and ~30% from
the 19-June peak, VKOSPI printed a record ~98 intraday, and realised vol (Bloomberg HV)
was **83**. Implied ATM (~68–82% depending on tenor) sits *below* realised — normal
right after a shock.

## 2. Data

Three option chains fetched via `bbg-fetch` (`data/fetch_option_chain.py`), reference
2026-07-24, spot 1055.58, plus the BVOL moneyness grid transcribed from the Option
Monitor screenshot.

| file | expiry | tenor | forward | rate (recovered) | r_eff = ln(F/S)/T | parity r² |
|---|---|---|---|---|---|---|
| `data/kospi2_20260813_20d.csv` | 13-Aug-2026 | 20d (0.055y) | 1053.11 | 16.47%* | −4.27% | 0.9996 |
| `data/kospi2_20260910_48d.csv` | 10-Sep-2026 | 48d (0.132y) | 1059.73 | 3.14% | +2.99% | 0.99987 |
| `data/kospi2_20261008_76d.csv` | 08-Oct-2026 | 76d (0.208y) | 1049.75 | 2.35% | −2.66% | 0.99992 |
| `data/bvol_moneyness_grid_20260724.csv` | — | 1W–2Y × 80–120% | — | — | — | — |

\* the 20d recovered rate is the ill-conditioned parity funding leg (disc ≈ 1 at 20
days); it is irrelevant to the vol shape. The forward is well identified at every tenor.

Each chain carries a commented `# key=value` header (spot, year_fraction, forward, rate,
r2, num_strikes_used) followed by the option table (bid/ask/last prices, Bloomberg
greeks, open interest). The forward is recovered by put-call parity regression.

## 3. Method

The pipeline reuses the **shipped** `goal_based_allocation.vanilla_option_pricer`
(Laplace-transform regime-switch pricer) without modification; the analysis code only
builds clean slices and drives calibration.

- **Clean OTM slice.** Out-of-the-money only (puts below forward, calls above), one price
  per strike, implied vols recomputed from price in **forward (undiscounted) space**, so
  the vol fit is independent of the funding rate. Primary price source is the
  parity-consistent `last`; `mid` is used where last prints are stale (short tenor), and
  off-quote last prints (last outside the bid/ask) are dropped. The market forward is
  matched exactly by setting the pricer carry to `r_eff = ln(F/S)/T`.
- **Mixture baseline.** A forward-constrained two-lognormal mixture `p·F₁+(1−p)·F₂ = F`,
  fitted in vega-weighted price space, with the risk-neutral density's skew, excess
  kurtosis and modes read off directly. A frown requires **negative excess kurtosis**
  (platykurtic, in practice bimodal); a skew/smile is leptokurtic.
- **Regime-switch calibration.** The shipped `RiskNeutralParams` (σ₀, σ₁, λ₀₁, λ₁₀, η₀, η₁)
  calibrated by multistart Nelder-Mead in vega-weighted forward-price space, from both
  start regimes. As a control, the pricer reproduces the framework's published
  shipped-model row to 0.1 vol pt.
- **BVOL check.** The interpolated BVOL grid compared, per strike, against the actual
  listed bid/ask implied vols at the nearest matching tenor (the 20d chain sits one day
  from the BVOL 3W column).

## 4. Results

### 4.1 The traded surface is a skew, not a frown

| tenor | ATM | frown depth (80/150) | frown depth (wings) | skew slope /logK | RND skew | RND excess kurt | modes |
|---|---|---|---|---|---|---|---|
| 20d (0.055y) | ~82 | +0.7 | −9.1 | −37 | −0.71 | **+1.35** | 1 |
| 48d (0.132y) | 74.0 | −0.7 | −1.8 | −15 | −0.45 | +0.67 | 1 |
| 76d (0.208y) | 70.9 | +0.0 | −0.4 | −7 | −0.14 | **+0.01** | 1 |
| framework digitised "frown" | ~72 | **+5.4** | — | ≈0 | +0.29 | −0.77 | 2 |

Every real tenor is leptokurtic, single-moded, and negatively skewed — the opposite of
the platykurtic, bimodal density a frown requires. The 20d slice adds a call-wing smile
(vol turns up in the far calls) on top of a steep put skew, but ATM is never bid over
both wings.

### 4.2 The shipped model fits it with a plain crash jump

The shipped exponential-jump model fits every tenor from the **growth** regime (where the
only jump is the downward crash, producing a put skew naturally):

| tenor | mixture RMSE | RS-growth RMSE | mean crash jump |
|---|---|---|---|
| 20d | ~1.3–2.3 | ~1.2–2.1 | ≈ −11% |
| 48d | 1.5 | 1.5 | ≈ −20% |
| 76d | 0.4 | 0.5 | ≈ −12% |

At the short end the growth fit **decisively beats** the stress fit (RMSE 1.2 vs 2.8) —
the steep short-dated put skew strongly prefers the crash-jump regime. On the digitised
frown the same model managed only RMSE 3.33 and could not lift ATM over the call wing; on
the real skew it needs no Erlang jump and no free-drift extension. The entire Erlang /
free-drift apparatus in the framework manufactures bimodality, and there is no bimodality
here to manufacture.

### 4.3 Term structure kills the frown signature

The framework's falsifiable §6 signature is a frown depth that turns positive and peaks at
+5.4 near 3M. The real term structure does the opposite: the ATM term structure is
**inverted** (short vol above long — a stress signature), the skew **flattens
monotonically** (slope −37 → −15 → −7), and the RND **Gaussianises** (excess kurtosis
+1.35 → +0.67 → +0.01, never crossing into the platykurtic region a frown needs). Frown
depth stays at or below zero at every tenor.

### 4.4 The frown is a BVOL surface artifact

In the liquid 80–120% band the BVOL 3M row is itself a mild downward skew (80% put +4.5
over ATM, 120% call −1.6). Compared per strike against the actual bid/ask at the matched
tenor (20d vs 3W), the interpolated surface sits **outside the tradeable market**: it
prints an ATM cell of 104 where the options quote 81–85, and marks the 81–95% put strikes
5–10 vol **below the bid**. The frown's entire left wall lives below 80% moneyness, where
the grid has **no node** — pure extrapolation — while the deep puts that actually trade
run the other way, rising to ~107 vol at the 59%-moneyness strike.

Two professional caveats sharpen rather than weaken this. Deep OTM puts are **not**
worthless at these vols (a 40%-moneyness 3M put is worth ~4 index points at 100% vol, ~60
at 200%), so their vol is informative and priced. And tradability is a **delta** band, not
a fixed-moneyness one: at 70–100% vol the 5-delta put sits near 50–60% moneyness, and in
this tape even a 1.4-delta put (the 59% strike) shows 12,437 lots of open interest. So the
wings are liquid and we can see they mark rich — the surface simply mis-draws them. At the
3M tenor itself the two-sided market is ~20 vol points wide, so no smooth curve — skew or
frown — is pinned by tradeable prices; asserting a crisp frown reads precision into noise.

### 4.5 Calibrated parameter term structure, and how it differs from the paper

Growth-regime fit (risk-neutral), per tenor, from the preserved pipeline. Dwell time is
1/λ in months; the expected jump is quoted in the paper's convention, E[e^J]−1. The
paper's object is the **balanced portfolio** — a diversified 40/40/20 bonds/equity/PE
mandate, not an equity index — so that is the correct reference; its standalone **equity**
asset is shown only as the closest asset-type analog to a single index.

| object | σ₀ | σ₁ | λ₀₁ (/yr) | λ₁₀ (/yr) | crash E[e^J]−1 | recovery E[e^J]−1 | growth dwell | stress dwell |
|---|---|---|---|---|---|---|---|---|
| KOSPI2 20d | 0.59 | 0.98 | 8.4 | 1.9 | −11% | +16% | 1.4 m | 6.4 m |
| KOSPI2 48d | 0.63 | 0.52 | 2.8 | 0.3 | −18% | +23% | 4.2 m | 44 m |
| KOSPI2 76d | 0.61 | 0.78 | 3.5 | ≈ 0 | −11% | +14% | 3.5 m | ∞ (absorbing) |
| **paper balanced portfolio (40/40/20)** | **0.11** | **0.16** | **0.10** | **1.00** | **−18%** | **+12%** | **120 m** | **12 m** |
| paper equity asset (Table 1) | 0.15 | 0.23 | 0.10 | 1.00 | −25% | +15% | 120 m | 12 m |

(KOSPI2 fit RMSE 2.1 / 1.5 / 0.5 vol pts at 20d / 48d / 76d.)

The comparison is contextual, not like-for-like — a single equity index in acute crisis
against a diversified multi-asset portfolio built for long-horizon allocation — but it
quantifies how far the snapshot sits from the paper's base case. Four things stand out.

**Diffusion vol is ~5–6× the balanced portfolio.** σ₀ ≈ 0.6 across tenors versus 0.11.
Part of that gap is single-index-versus-diversified (the balanced portfolio is 40% bonds
and pools three imperfectly-correlated assets); but even the paper's standalone equity
asset (0.15) sits ~4× below the KOSPI crisis level (realised HV 83).

**Crash intensity is 30–80× the paper.** λ₀₁ ≈ 3–8 /yr versus 0.10 /yr: the paper prices
a crash once a decade, the KOSPI snapshot prices it as arriving several times a year, so
the growth regime dwells 1–4 months rather than 10 years — a live, near-continuous crash
regime.

**The jump size is not extreme.** Expected crash −11% to −18%, against the balanced
portfolio's −18% (and the equity asset's −25%) — comparable, even a touch smaller. This
matches the study's core finding: the steep skew is bought with crash *frequency* and
diffusion vol, not with one large jump, which is exactly why the plain exponential jump
suffices and no Erlang shape is needed.

**Recovery inverts.** The paper has λ₁₀ = 1.0 (mean-reverting, ~1-year stress). The
calibrated λ₁₀ falls toward 0 by 76d — over the option horizon the market prices a crash
it falls into and does not quickly climb out of.

Two caveats on reading this. The paper's calibration is real-world (P-measure) for
allocation (portfolio growth drift ~4.2%, ~10-year cycle); the KOSPI numbers are
risk-neutral (Q), from a single stressed option snapshot, so part of the λ and vol gap is
the risk premium on top of the crisis. And one slice under-identifies the σ₀/σ₁/λ split —
growth vol, crash intensity and jump size trade off along a ridge of near-equal RMSE, so
read the term structure as indicative. The robust features are stable: high vol, high
crash intensity, a moderate jump size, and a vanishing recovery intensity.

### 4.6 Which starting regime, and the smile viewed from each

Every slice is calibrated from **both** starting regimes; the reported parameters are the
**regime 0 (growth)** fit. The two regimes differ by the direction of the *pending* jump,
not by their vol level: from regime 0 the next transition is the downward crash (0→1),
from regime 1 it is the upward recovery (1→0). A put skew needs a pending down-jump, so it
can only be produced from regime 0 — which here calibrates to ~60% diffusion vol, i.e. a
high-vol state braced for a crash, not a calm one.

`figures/kospi2_model_fit_by_regime.png` prices the *same* calibrated parameters from each
starting regime against the market bid/ask. From regime 0 the model reproduces the market
put skew and sits inside the bid/ask — this is the fit. From regime 1 the same parameters
give a different, non-fitting shape: flat at 76d (there λ₁₀ ≈ 0, so from stress the chain
never leaves and the smile is a pure σ₁ lognormal) and recovery-tilted at the shorter
tenors, at the wrong level. Regime 0 is the only starting state consistent with a
downside-skewed market. (Reproduce with `term_structure.plot_model_fit_by_regime()`.)

## 5. Analytics delivered

- Parity-recovered forward/rate diagnostics and a clean OTM implied-vol slice builder for
  bbg-fetch chains, discount-invariant in forward space, with stale-quote filtering.
- Forward-constrained two-lognormal mixture fit with full risk-neutral moment extraction
  (skew, excess kurtosis, modes) — the model-free benchmark and the frown/skew classifier.
- Calibration of the shipped regime-switch jump-diffusion (both start regimes) to each
  chain, reusing the shipped Laplace pricer, with implied crash/recovery jumps and
  martingale drifts.
- A three-tenor term-structure of frown depth, skew slope, and RND kurtosis, tested
  against the framework's §6 prediction.
- A per-strike comparison of the BVOL interpolated surface against listed bid/ask vols,
  localising the "frown" to interpolation error plus off-grid extrapolation.
- Delta-space liquidity and deep-put valuation checks.
- A calibrated-parameter term structure (σ, λ, η, drifts, dwell times per tenor) set
  against the paper's balanced-portfolio (40/40/20) effective dynamics, quantifying the
  crisis: ~5–6× the diffusion vol and 30–80× the crash intensity, at a comparable jump
  size.

## 6. Figures

| file | shows |
|---|---|
| `figures/kospi2_skew_vs_frown.png` | 48d slice + mixture/RS fits, and the real skew vs the digitised frown |
| `figures/kospi2_term_structure.png` | 48d vs 76d skew flattening; frown-depth vs the §6 prediction |
| `figures/kospi2_term_structure_3tenor.png` | 3-tenor: fanning skew, frown depth vs prediction, kurtosis decay |
| `figures/kospi2_traded_vs_bvol_simple.png` | traded mid + bid/ask vs nearest BVOL tenor, per maturity |
| `figures/kospi2_three_views.png` | the fitted BVOL curve vs the OMON grid vs listed quotes at ~3M |
| `figures/kospi2_frown_vs_traded.png` | the frown = liquid skew + extrapolated wings; deep-put sign flip |
| `figures/kospi2_interpolation_misleading.png` | interpolated surface vs actual bid/ask; off-grid wings; wide 3M market |
| `figures/kospi2_model_fit_by_regime.png` | model fit vs market bid/ask, smile viewed from regime 0 (growth) vs regime 1 (stress) |

## 7. How to run

Requires `goal-based-allocation` installed (this repo) plus `numpy`, `scipy`,
`pandas`, `matplotlib`.

```bash
cd paper_code/kospi_volatility_fit_jun2026
python run_analysis.py                 # full term structure + 3-tenor figure
```

Or drive individual cases via the dispatcher:

```python
from run_analysis import LocalTest, run_local_test
run_local_test(LocalTest.SLICE_ONE)      # one clean slice + parity check
run_local_test(LocalTest.MIXTURE)        # mixture baseline + RND moments
run_local_test(LocalTest.REGIME_SWITCH)  # shipped RS model, both regimes
run_local_test(LocalTest.TERM_STRUCTURE) # three-tenor diagnostics + figure
run_local_test(LocalTest.BVOL_CHECK)     # BVOL grid vs listed bid/ask
```

### Module map

| module | role |
|---|---|
| `vol_surface_utils.py` | chain parsing, forward-space BS/IV, OTM slice, lognormal mixture + RND moments |
| `regime_switch_calibration.py` | calibrate the shipped `RiskNeutralParams` to a slice, both regimes |
| `term_structure.py` | per-tenor diagnostics, three-tenor table and figure |
| `bvol_interpolation_check.py` | BVOL fitted grid vs listed bid/ask, per strike |
| `run_analysis.py` | `LocalTest` enum + `run_local_test` dispatcher |

## 8. Caveats

- Three tenors on a single reference date — indicative, not a full simultaneous
  1M/3M/6M/1Y strip. The qualitative conclusions (skew not frown; growth-regime fit; no
  Erlang needed) are robust across price sources; exact RND moments and RMSE move a vol
  point or so with the price source and mixture restart count (the pipeline uses a fixed,
  documented configuration: `mid` at 20d, `last` elsewhere, 60 mixture restarts).
- The absolute vol level (~70–82% ATM) is crisis-high but corroborated by HV 83 and the
  ~30% drawdown; it is not a data error.
- The BVOL grid is transcribed by hand from a screenshot and contains visible vendor fit
  artifacts (the 100% column spikes to 104/92 at 3W/1M; the 9M–2Y rows are scattered
  noise). These are retained as-is because their unreliability is part of the finding.
