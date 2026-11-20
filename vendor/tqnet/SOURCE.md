# TQNet source

The authors' [TQNet repository](https://github.com/ACAT-SCUT/TQNet) accompanies
[Temporal Query Network for Efficient Multivariate Time Series Forecasting](https://arxiv.org/abs/2505.12917).
Files were downloaded from the official archive at commit
[`15e19cb23483ed52398566c4baa959168cfffa57`](https://github.com/ACAT-SCUT/TQNet/commit/15e19cb23483ed52398566c4baa959168cfffa57).
The archived bytes were checked against GitHub's repository contents API at that commit.

| Local path | Upstream path | SHA-256 |
| --- | --- | --- |
| `TQNet.py` | `models/TQNet.py` | `c31e4498104d241a69335d22522f01a28cae90d58d73f4ec4c10929fb53b7727` |
| `LICENSE` | `LICENSE` | `50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a` |

The vendored files are unchanged and used under Apache 2.0. The project adapter
in `src/tqnet_adapter.py` sets a one-step channel forecast, hidden width 32,
0.1 output dropout, no RevIN, and accepts the cycle selected from training-only ACF. A trainable
linear head maps all feature channels to one return estimate. Cycle positions
follow the upstream data loader's exclusive-end convention and are computed
from the signal-day trading ordinal.
