# Single MaxCon

This experiment is the max-concurrency counterpart to `single-qps`.

Instead of overriding replay launch timing with a Poisson QPS target, this
workflow keeps the compiled launch pattern and only overrides
`launch_policy.max_concurrent`.

Use the local generator here:

- `experiments/single-maxcon/local/generate_experiment.py`

Runbook:

- `experiments/single-maxcon/local/README.md`
