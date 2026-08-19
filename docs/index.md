# GoalBasedAllocation

`goal-based-allocation` provides analytical dynamic mean-variance allocation and
terminal-wealth risk under regime-switching jump-diffusions for quantitative researchers and
wealth-management model developers.

It solves a two-regime model with exponential jumps at regime transitions and an absorbing wealth
floor. Multi-asset mandates are aggregated to one effective risky asset. Monte Carlo is used to
validate the analytical Laplace-transform and Riccati calculations, not to implement them.

Start with the [wheel-first quickstart](getting-started.md), then read the
[conventions](conventions.md) before interpreting a numerical result.

```{toctree}
:maxdepth: 2
:caption: Start here

getting-started
conventions
model-boundaries
```

```{toctree}
:maxdepth: 2
:caption: Allocation workflows

user-guide/mv-optimal-policy
user-guide/terminal-wealth-floor
user-guide/mandates-opportunity-set
```

```{toctree}
:maxdepth: 2
:caption: Supporting workflows

user-guide/option-pricing
validation
papers
comparison
api/index
```

## Project links

- [PyPI](https://pypi.org/project/goal-based-allocation/)
- [Source](https://github.com/ArturSepp/GoalBasedAllocation)
- [Issues](https://github.com/ArturSepp/GoalBasedAllocation/issues)
- [Changelog](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CHANGELOG.md)
- [Paper (SSRN 6534579)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6534579)

This software is research code distributed without warranty and does not provide investment
advice.
