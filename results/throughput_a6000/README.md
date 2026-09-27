# Generation throughput, NVIDIA RTX A6000

Measured on 2026-09-25 on an RTX A6000 48GB with `scripts/benchmark_throughput.py`, run from a medlatents checkout (import paths differ from the copy in this repository, the timing logic is identical; `meta.script_sha256` records the exact file that ran). These numbers back the throughput table in the paper (`tab:throughput`).

| File | Generators | Repeats |
|---|---|---|
| `throughput_fast.json` | AR, MaskGIT, DFM, tuned D3PM, tuned SEDD, RF | 3 |
| `throughput_slow.json` | D3PM, SEDD (1000 steps) | 1 |
| `throughput_ldm.json` | LDM (1000 steps) | 1 |
| `throughput_bfn.json` | BFN (1000 steps) | 1 |

All runs: 10K samples, float32 without autocast, randomly initialised weights (cost per step does not depend on them), vocabulary 1024 (LFQ-1024 for discrete, VAE-c4 for continuous), one warm-up batch before timing. The BFN sampler masks the pad and mask tokens before the softmax, as medlatents does from 0.1.2; with random weights, sampling without that mask fails at step 968.
