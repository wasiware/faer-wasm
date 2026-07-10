# SVD + eigenvalue wasm-shaping research — focused pass (2026-07-10)

Third research pass in the series (after LU, QR). **Method: the "focused"
tier** — a single research subagent (6 web searches + 3–4 primary/
authoritative sources), ~2 min / ~35k tokens, then verified by hand
against Opus's own knowledge of the LAPACK/MAGMA literature. Not the
100-agent deep workflow, not a bare web crawl. Grades below carry that
provenance: structural algorithm facts are **observed/by-hand** (published
properties + definitive OpenBLAS source, cross-checked by hand), not the
3-vote adversarial panel.

## The one finding that shapes everything

**The eigen flank is dominated by the Householder *reduction*, not the
back-end** — and all three targets are the same kind of reduction:

| target | reduction | level-2 share (memory-bound) | back-end |
| - | - | - | - |
| SVD | `dgebrd` (bidiagonalize) | **~50%** (`dgemv` panels, `dlabrd`) | `dbdsqr` / `dbdsdc` |
| symmetric eig | `dsytrd` (tridiagonalize) | **~50%** (`dsymv`) | `dstedc` (D&C) |
| general eig | `dgehrd` (Hessenberg) | **~20%** (rest is gemm) | `dhseqr`/`lahqr` |

Reduction is "typically the bottleneck… up to 70% of total time," and at
n≤512 the level-2 panel share is *higher* still relative to the O(n³) gemm
back-end. Sources: ICL/Dongarra SVD-bidiag (icl-utk-953-2017), MAGMA
memory-bound kernels (icl-utk-530-2012), STFC symmetric-diagonaliser
report (DLTR-2007-005). This is the *same disease* as LU/QR — a
memory-bound level-2 panel bottlenecking an otherwise-cubic algorithm —
which is exactly the shape our flat simd128 kernels win.

## Confirmed (authoritative source + by-hand check)

1. `dgebrd` (SVD) and `dsytrd` (sym-eig) each spend **~50% of flops in
   Level-2 BLAS** (`dgemv` / `dsymv` respectively); two-sided reductions
   have far heavier panels than one-sided LU/QR. `dgehrd` (gen-eig) is the
   outlier at **~80% Level-3** (Quintana-Ortí/van de Geijn restructuring) —
   much more gemm-friendly.
2. **OpenBLAS optimizes NONE of the eig/SVD routines.** Its `lapack/` dir
   is exactly `getf2 getrf getrs laed3 laswp lauu2 lauum potf2 potrf trti2
   trtri trtrs` — LU + Cholesky + triangular, **plus one outlier `laed3`**.
   Absent: `gebrd sytrd gehrd labrd latrd lahr2 bdsqr bdsdc stedc hseqr
   lahqr larfb larf symv`. So Pyodide/scipy runs **reference-netlib
   LAPACK over generic-C BLAS** for every SVD/eig reduction and back-end —
   the *same weak opponent we beat 2.5–3× on QR.* (github OpenMathLib/
   OpenBLAS/tree/develop/lapack — source of truth.)
3. The lone optimized outlier `laed3` is the **gemm-bound eigenvector-
   forming kernel** of symmetric D&C — confirming the D&C back-end's bulk
   is a dense matmul (route to faer gemm), not something to hand-shape.

## Unverified / open (cheap to close ourselves)

- The reduction-vs-back-end time split **at n≤512 specifically on wasm** —
  not in the literature; one profiling run on our own harness settles it.
- **Does faer's own `gebrd`/`sytrd`/`gehrd` already block into its gemm
  well, or does it need the unblocked-panel workaround we applied to
  Schur?** This is the pivotal question for *effort*: if faer's reductions
  are native-shaped (assume wide SIMD) and lose on wasm like its Schur did,
  the lever may be **param-tuning (faer-schur style)** rather than a
  from-scratch flat panel. Check `faer-rs/` directly + measure.
- Whether faer exposes `symv` / `syr2k` publicly (gates the sym-eig plan).

## Ranked levers (most leverage first)

1. **Flat simd128 matrix-vector reduction panel + block-apply→faer-gemm**
   (`dgemv`/`dlarfb` family). Unblocks **SVD + general-eig together** —
   the single highest-coverage kernel. For `gehrd` (80% L3) the win is
   mostly routing the update into faer's fast gemm; for `gebrd` (50% L2)
   the flat simd128 GEMV panel is the differentiator. Expected vs scipy:
   **large** (generic-C reference opponent, as with QR).
2. **Flat simd128 `dsymv` + `dsyr2k` → the `dsytrd` panel.** Unblocks
   **symmetric eig.** The one genuinely distinct kernel (exploits
   symmetry — reads one triangle, updates via rank-2k not general gemm).
   `dsytrd`'s SYMV is provably ~50% of runtime and fully generic in scipy →
   **large** payoff for symmetric problems; build second.
3. **Shared block-apply `(I − V Tᵀ Vᵀ)` over faer gemm** — connective
   tissue for levers 1–2, mostly plumbing.
4. **Leave the back-ends alone** (`bdsqr`/`stedc`/`hseqr`): a *decision*,
   not a build. D&C bulk (`laed3`) is already gemm-shaped; QR-iteration
   back-ends are low-flop scalar loops at n≤512; the `lahqr` Schur fix is
   already shipped. Don't invest until profiling says otherwise.

## Net

Sub-question-4 thesis holds: **one wasm-shaped Householder-reduction
family serves SVD and general-eig at once**, with symmetric-eig needing
one extra symmetric kernel pair. And because OpenBLAS optimizes none of
it, scipy is running generic-C reference LAPACK across the board — so
shaped reduction kernels should **win, not merely close the gap.**

**Before building, close the one pivotal open question** (tune faer's
reductions vs build flat panels) — same fork QR faced, where faer's
`block_size=1` param already won and the kernel then beat *that*. Measure
first; it decides whether this is a tuning job or a kernel job.
