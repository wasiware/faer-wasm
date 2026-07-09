# faer-wasm vs Pyodide (scipy/numpy on wasm) — 2026-07-09

The first external head-to-head, per the architect's benchmark direction:
the same problems solved by the incumbent scientific-computing-on-wasm
stack, in the same V8, both sides single-threaded and paying the same
wasm tax. Run on a GitHub Actions runner (`pyodide-bench.yml`,
`bench/pyodide-vs-faer.mjs`): faer bench-harness wasm (opt-level 3,
carried patches, `faer-schur` for Schur) vs Pyodide 314.0.2 with
numpy 2.4.3 / scipy 1.18.0 under node v22.23.1. Adaptive min-of-3 per
measurement; n = 64 / 128 / 256.

## Headline — honest version

**Pyodide wins most of this suite.** Geomean speedup of faer-wasm over
Pyodide: **0.41×** — i.e. scipy/numpy's reference LAPACK, f2c'd to wasm,
is ~2.4× faster overall at these sizes. faer-wasm wins exactly where its
hand-written gemm microkernels dominate (matmul, 4–5× at n ≥ 128, real
and complex); it loses the factorizations and eigensolvers, typically
2–10×.

| op | n | faer-wasm | pyodide | speedup |
| - | -: | -: | -: | -: |
| matmul | 64 | 0.44 ms | 0.25 ms | 0.6× |
| lu_solve | 64 | 1.01 ms | 0.09 ms | 0.1× |
| qr_r | 64 | 3.64 ms | 0.21 ms | 0.1× |
| svd | 64 | 11.54 ms | 1.65 ms | 0.1× |
| eigvals | 64 | 3.12 ms | 1.50 ms | 0.5× |
| schur | 64 | 10.44 ms | 1.68 ms | 0.2× |
| matmul_c64 | 64 | 0.84 ms | 0.41 ms | 0.5× |
| lu_solve_c64 | 64 | 1.77 ms | 0.19 ms | 0.1× |
| qr_r_c64 | 64 | 3.95 ms | 0.81 ms | 0.2× |
| schur_c64 | 64 | 13.09 ms | 5.42 ms | 0.4× |
| matmul | 128 | 0.82 ms | 3.24 ms | **4.0×** |
| lu_solve | 128 | 3.21 ms | 0.56 ms | 0.2× |
| qr_r | 128 | 7.77 ms | 1.31 ms | 0.2× |
| svd | 128 | 20.71 ms | 9.13 ms | 0.4× |
| eigvals | 128 | 116.80 ms | 11.44 ms | 0.1× |
| schur | 128 | 24.91 ms | 14.18 ms | 0.6× |
| matmul_c64 | 128 | 2.47 ms | 3.45 ms | **1.4×** |
| lu_solve_c64 | 128 | 3.82 ms | 1.29 ms | 0.3× |
| qr_r_c64 | 128 | 12.06 ms | 5.85 ms | 0.5× |
| schur_c64 | 128 | 37.74 ms | 36.79 ms | 1.0× |
| matmul | 256 | 5.98 ms | 30.86 ms | **5.2×** |
| lu_solve | 256 | 13.51 ms | 4.11 ms | 0.3× |
| qr_r | 256 | 42.68 ms | 10.06 ms | 0.2× |
| svd | 256 | 127.39 ms | 68.31 ms | 0.5× |
| eigvals | 256 | 227.77 ms | 79.40 ms | 0.3× |
| schur | 256 | 209.27 ms | 88.12 ms | 0.4× |
| matmul_c64 | 256 | 18.98 ms | 84.53 ms | **4.5×** |
| lu_solve_c64 | 256 | 21.10 ms | 9.61 ms | 0.5× |
| qr_r_c64 | 256 | 66.94 ms | 49.80 ms | 0.7× |
| schur_c64 | 256 | 376.70 ms | 278.01 ms | 0.7× |

## Reading it

- **The pattern is one story told twice.** faer's kernels are shaped for
  native machines: wide SIMD, deep blocking, cache-oblivious recursion.
  On wasm those shapes become overhead — this is the same disease as
  every blocking cliff this repo has measured, now visible externally.
  Reference LAPACK is "dumb" Fortran through f2c: simple loops that
  compile to lean wasm and, at n ≤ 256, win.
- **matmul is the counterexample that proves it**: the gemm crate's
  microkernels are the one place faer's approach carries to wasm, and
  there faer beats numpy's wasm matmul 4–5×. That gap is also the
  long-game lever — LAPACK's own blocked factorizations bottom out in
  dgemm, ours in the fast gemm, so the crossover should flip toward
  faer as n grows past this table. Unmeasured; next question.
- **The §7 tuned kernels close some gap but not all of it**: tuned LU at
  n=128 is ~0.24× of the default (≈ 0.77 ms) vs Pyodide's 0.56 ms; QR
  panel-1 ≈ 1.9 ms vs 1.31 ms. The comparison above uses faer's
  *default* high-level calls, which is what a naive consumer gets.
- **eigvals @128 (0.1×) is faer's `.eigenvalues()` blocking cliff**
  (`benchmarks-2026-07.md`) — no public params to fix it; `faer-schur`'s
  tuned Schur is the workaround and scores 0.6× rather than 0.1×.

## Caveats

- Each side does its idiomatic call for the same problem (e.g. QR is
  R-only on both sides via `mode='r'`; SVD computes full U/V on both).
- One run, one shared runner, min-of-3 — treat entries within ~±30% as
  ties; the 3×+ gaps are real.
- Pyodide numbers include no marshalling: timing loops run entirely in
  Python inside the wasm VM, as ours run entirely in wasm.
- n ≤ 256 only (leak-capped harness). The matmul advantage says the
  large-n story likely reverses; not yet evidence.

## Follow-ups recorded in ROADMAP

1. Extend to n = 512/1024 for the crossover (factorizations ride gemm).
2. Re-run with §7-tuned faer calls to quantify the honest best-case.
3. The strategic option nobody has taken: faer's algorithms with
   wasm-shaped parameters everywhere (what `faer-schur` did for Schur).

## Regenerate

Actions → "pyodide bench" → Run workflow (manual; the dev container
blocks the Pyodide CDN, runners don't). Results: job log + artifact.
