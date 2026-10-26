| Method | Proxy score % | True held-out accuracy % [95% CI] | Exploit rate % | Gap Δ (pp) | Training overhead %: optimizer / rollout† |
|---|---:|---:|---:|---:|---:|
| Baseline: outcome + penalty | 99.87 | 0.11 [0.10, 0.13] | 99.785 | 99.76 | +0.0 / +0.0 |
| A — Rta projection | 99.86 | 0.15 [0.13, 0.17] | 99.712 | 99.71 | +11.6 / +1.3 |
| B — Nyaya verification | 99.96 | 99.88 [99.87, 99.88] | 0.098 | 0.09 | +2.7 / +566.2 |
| C — process validity | 38.09 | 25.22 [20.72, 29.95] | 4.402 | 12.86 | +3.9 / +6.3 |
| D — context hierarchy | 99.84 | 99.80 [99.78, 99.81] | 0.000 | 0.04 | -0.1 / +32.1 |
| E — unified synthesis | 99.88 | 99.86 [99.85, 99.88] | 0.000 | 0.01 | +4.4 / +625.5 |
