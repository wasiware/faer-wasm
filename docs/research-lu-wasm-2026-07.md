# LU optimization research — general + wasm-shaping (2026-07-09)

Deep-research pass ordered by the architect after the first shaped LU
kernel landed (beats scipy at n=64–128, 1.35–1.5× behind at 256–512).
Method: 5-angle web search fan-out → 15 sources fetched → falsifiable
claims extracted → 3-vote adversarial verification per claim. The
verification stage was cut short by a session usage limit, so claims
are graded **confirmed** (survived 3-0 or 2-1 adversarial votes),
**sourced-unverified** (extracted with quotes, votes never ran), or
**refuted** (killed 0-3/1-2). Per the working contract, nothing below
is graded above its evidence.

## Confirmed (adversarially verified)

1. **Recursive (Toledo) LU is the canonical fix for exactly our
   bottleneck.** The panel factorization is the memory-bound step with
   a sequential pivot dependency, in contrast to the compute-bound
   trailing update [LAWN 259, 2-1]. Formulating the panel recursively
   "casts memory-bound kernels into Level 3 BLAS operations"
   [LAWN 259, 3-0] — i.e., our measured ~3 GF panel wall is the known
   disease and recursion is the known cure.
2. **The I/O math favors it asymptotically**: Toledo's recursive LU
   performs Θ(√M/n) fewer cache misses than right-looking blocked LU
   [SIAM SIMAX 10.1137/S0895479896297744, 3-0], and empirically beat a
   similarly-coded right-looking LU on six RISC architectures [same,
   3-0].
3. **It is a drop-in with identical pivoting semantics**: LAPACK ships
   it as `SRC/VARIANTS/lu/REC` (iterative Toledo) [lapack 3.3.1 docs,
   3-0], claimed within 2× of the optimal number of memory transfers
   for square matrices [same, 3-0].
4. **The base-case crossover is a first-class tunable**: ReLAPACK
   switches to unblocked kernels below a configurable crossover
   "to avoid tiny BLAS-3 routines", settable per routine
   [ACM TOMS 10.1145/3061664, 3-0]. This matches our measured
   pure-panel-≤128 finding.

## Refuted (do not build on these)

- "ReLAPACK's recursive LU outperforms OpenBLAS and MKL" — **0-3**.
  Recursion fixes the panel/gemm imbalance; it does not beat
  vendor-tuned stacks. Expectation-setting: our win condition is that
  recursion feeds *our* gemm (measured 4.7 GF in context, faster than
  Pyodide's), not that recursion is magic.
- "Recursion is exactly four building blocks per split, all flops at
  rank n/2" — **1-2**; the real dgetrf2 recursion is messier (pivot
  application interleaves, ranks vary).

## Settled since (2026-07-09, post-research)

- **The Pyodide BLAS claim is now TESTED, no longer unverified** —
  `pyodide-vs-faer.mjs` prints `numpy/scipy.show_config()` on every
  run (durability: scripted). Run 4 (29023029694) shows scipy 1.18.0
  linked against **OpenBLAS 0.3.28, target `RISCV64_GENERIC`**
  (`... NO_AFFINITY=1 USE_OPENMP= RISCV64_GENERIC MAX_THREADS=4`),
  cross-compiled by Emscripten. Generic C kernels confirmed; the
  strategic consequence (route flops into our gemm) is confirmed with
  it.
- **Plan item 1 executed** (`lu_factor_recursive_in_place`,
  kernels/src/lu.rs): the projection held — 22.40 ms at n=512 on the
  runner (projected 20–22; blocked wk driver 27.39, scipy 18.45), and
  2.78 ms at n=256 (wk 3.26, scipy 2.44). One *constant* from the
  theory was refuted by measurement: narrow base cases (crossover
  8–32, trsm base 32) LOSE — skinny gemm call overhead exceeds the
  flat-loop cost it replaces. Winning shape: crossover 128, trsm
  base 64, flat loops alone through n=128. See
  docs/benchmarks-vs-pyodide-2026-07.md Run 4.

## Sourced-unverified (quotes extracted; adversarial votes never ran)
- **OpenBLAS's own getrf is recursion-shaped**: first panel width =
  min(m,n)/2, rounded to the gemm microkernel width and capped by the
  gemm cache parameter, falling back to unblocked getf2 only at tiny
  widths. Production practice agrees with the confirmed theory.
- **LAPACK's dgetrf2**: recursive split at n1 = min(m,n)/2 → factor
  left, pivot, trsm A12, gemm A22, recurse right.
- **Wasm ceilings**: simd128 is a hard 2-lane f64 cap; relaxed-simd
  adds f64x2 madd/nmadd (FMA expression, fusion not guaranteed —
  engine-dependent); production runtimes bounds-check via guard pages
  (cheap) with software checks only as fallback (up to ~650% worst
  case, not the V8-on-x86-64 configuration we target).
- LU elimination steps cost half the flops of QR steps in tiled
  solvers (why LU-first matters); BLIS dgemm has been ported to
  browsers via Emscripten (blis-bench) — prior art exists but no
  measured-efficiency numbers were captured before the cutoff.
- Pivoting alternatives (RBT/no-pivot + one-step refinement, GERCP
  randomized complete pivoting, LU_PRRP) — all extracted, none
  verified. All change semantics vs LAPACK getrf; parked unless the
  structural fixes stall.

## Ranked plan for the 1.35–1.5× gap at n=256–512

1. ~~**Recursive panel (dgetrf2-shape)**~~ — **done** (see "Settled
   since" above): gap now 1.14×/1.21× at 256/512. Original rationale:
   the only technique that is confirmed, drop-in, and aimed at our
   exact measured wall; flop accounting projected n=512 at ~20-22 ms
   vs current 27.7 and scipy's 18.4. Measured: 22.40.
2. **Relaxed-FMA variants of the base-case kernels** (axpy →
   relaxed_madd) for the relaxed build — additive, small, rides
   infrastructure we already gate. Now the top open item for LU.
3. ~~**Verify the Pyodide-OpenBLAS claim ourselves**~~ — **done**,
   printed by every pyodide-bench run; OpenBLAS 0.3.28
   `RISCV64_GENERIC` confirmed.
4. **Pivoting alternatives** — parked: unverified, semantics-changing,
   and (1) closed most of the gap; the eigen flank outranks the LU
   residual now.

## Regenerate / extend

Workflow run wf_ad53c449-b3f (57/105 agents completed before the usage
limit; journal in the session transcript dir). Re-run the deep-research
workflow with the same args to re-verify the unverified pool.
