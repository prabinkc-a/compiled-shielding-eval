| Method | Small monitored accuracy % | Blind-spot survival / true accuracy % [95% CI] | Sampled no-exploit survival % | Sampled exploit rate % | Calibrated training CPU / rollout CPU × baseline |
|---|---:|---:|---:|---:|---:|
| Baseline GRPO | 8.33 | 30.15 [17.26, 50.81] | 30.197 | 69.803 | 1.00× / 1.00× |
| D: hierarchy | 99.89 | 90.31 [86.55, 93.93] | 90.875 | 9.125 | 1.12× / 1.18× |
| E: static synthesis | 99.89 | 90.31 [86.55, 93.93] | 90.875 | 9.125 | 1.30× / 1.64× |
| F: adaptive synthesis | 99.85 | 99.90 [99.83, 99.98] | 100.000 | 0.000 | 1.33× / 1.44× |
