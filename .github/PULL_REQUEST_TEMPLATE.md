## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact commands run and their results. For numerical changes, name the independently
implemented analytical identity or Monte Carlo cross-check.

## Checklist

- [ ] Tests cover the changed behavior or defect.
- [ ] Rate, annualisation, wealth, floor, regime, jump, and horizon conventions remain explicit.
- [ ] Analytical changes were checked independently; tolerances and expected values were not fitted.
- [ ] The published model specification and function-based public API remain intact.
- [ ] No runtime dependency beyond NumPy, SciPy, and Matplotlib was added.
- [ ] `papers/`, paper values, generated figures, private data, and local environments are unchanged.
- [ ] `uv run --no-sync pytest -q` and the relevant lint/docs/artifact checks pass.
- [ ] User-visible changes are documented in `CHANGELOG.md` and relevant docs.
- [ ] Public-signature, default, dependency-floor, or runtime-dependency changes are called out explicitly.
