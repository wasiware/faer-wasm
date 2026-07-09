# QR optimization research — general + wasm-shaping (2026-07-09)

Deep-research pass ordered by the architect, second in the series after
the LU pass (`docs/research-lu-wasm-2026-07.md`). Same harness: 5-angle
web-search fan-out → 14 sources fetched → 36 falsifiable claims extracted
(4 primary sources contributed: Schreiber & Van Loan's compact-WY paper,
the 2006 UT-transform TOMS paper, an arXiv HT hardware-codesign paper, and
the netlib LAPACK/OpenBLAS primary sources).

**Verification provenance — read this before trusting the grades.** The
3-vote adversarial panel that graded the LU pass **did not run here**: it
died mid-phase when the Fable 5 session exhausted its usage credits (all
25×3 verifier agents returned the credit error, 0 valid votes). Rather
than re-run the whole fetch (which had also partially hit the limit),
verification was done **by hand on Opus 4.8 against primary-source
knowledge of the LAPACK/OpenBLAS codebase**. That is a real check for the
structural claims (they are the shipped Fortran, stable across versions,
which Opus knows directly) but it is *not* the independent multi-vote
adversarial check the LU claims got. Grades below carry that caveat in
their durability axis: LAPACK-structure claims are **proven / by-hand**;
single-paper performance numbers are **observed / sourced-unverified**.
The pass is re-runnable (see Regenerate) once credits reset, to lift the
structural claims to cross-checked and actually vote the performance ones.

## Confirmed (checked by hand against primary source — LAPACK/OpenBLAS)

These are structural facts about the shipped reference LAPACK and OpenBLAS
code. Strength **proven** (they are the code), durability **by-hand**
(confirmed from Opus's knowledge of the primary sources, not re-fetched or
tested in-repo).

1. **The incumbent we race on QR is *doubly* un-optimized.** OpenBLAS's
   own optimized-LAPACK layer replaces only LU / Cholesky / triangular-
   inverse (`getrf`, `getf2`, `getrs`, `laswp`, `potrf`, `potf2`,
   `lauum`, `trtri`, …) — **it ships no QR routines**. So Pyodide's scipy
   QR is *reference netlib* `dgeqrf`/`dlarft`/`dlarfb` Fortran, layered
   over OpenBLAS's **generic-C `RISCV64_GENERIC` BLAS** (confirmed by
   `show_config()` in the LU pass). Two strikes: no arch BLAS microkernels
   *and* no optimized QR structure. This is why `qr_r_tuned` already beats
   scipy 1.3–1.7× at every measured size — and it means the margin is
   structural, not luck. (Contrast LU, where scipy runs OpenBLAS's *own*
   `getrf` — a tougher opponent, which is exactly why our LU wins were
   narrower.)
2. **Reference LAPACK `dgeqrf` is right-looking blocked**: loop over
   panels of width `IB = min(K−I+1, NB)`; unblocked `dgeqr2` factors each
   panel; `dlarft` forms the compact-WY triangular factor `T`; a single
   `dlarfb` applies `H^T` to the trailing block (the only large-rank
   gemm-bearing step). `T`-formation and block-apply run **only when a
   trailing matrix exists** — the last panel skips both.
3. **LAPACK itself switches to unblocked below a crossover.** `dgeqrf`
   queries `NX = ILAENV(ISPEC=3)`, runs the blocked loop only up to
   `K−NX` columns, and factors the tail (and the *whole* matrix when
   `min(M,N) ≤ NX`) with unblocked `dgeqr2`. The blocked path is entered
   only if `NB ≥ NBMIN` (floor 2), `NB < K`, and `NX < K`. **The
   reference implementation's own control logic encodes that blocking
   loses at small n** — precisely our measured regime.
4. **Panel width degrades gracefully with workspace**: optimal needs
   `LWORK ≥ N·NB`; with less, `NB` drops to `LWORK/N`, falling to
   unblocked only below `NBMIN`. Block width is a soft knob, not a hard
   strategy switch.
5. **Compact WY (`Q = I ± Y T Y^T`, `T` upper-triangular r×r) is exactly
   what `dlarft` forms and `dlarfb` applies** — the mechanism that routes
   block-application into matrix-matrix multiply. Its `T` workspace is
   only ~r²/2 entries, negligible against the panel. (Schreiber & Van
   Loan 1989; storage win over the original Bischof–Van Loan `W` form.)
6. **Production LAPACK ships a fully recursive QR, `dgeqrt3`** (added 3.4.0,
   2011-11-11, contributed by R. James; based on Elmroth–Gustavson 2000):
   splits `N1 = N/2`, factors the left half recursively, applies `Q1^T` to
   the right through `DTRMM`+`DGEMM` of rank `N1`. **Its base case is a
   single column (`N=1`, one `dlarfg`) with no coarse crossover** — so the
   bottom recursion levels issue inner-dimension 1,2,4,… TRMM/GEMM calls:
   *exactly the skinny-gemm regime the LU pass measured as a wasm loss.*
7. **LAPACK 3.4.0 also added the sequential CAQR / tile kernels**
   (`xGEQRT`, `xTPQRT`/`xTPMQRT`/`xTPRFB` — "triangle on top of pentagon")
   with the block size exposed in the interface rather than hidden in
   `ILAENV`. `xGEQRT` caches the compact-WY `T` matrices explicitly so a
   later `xGEMQRT` can re-apply `Q` cheaply.

## Refuted / do-not-build (this pass's most important negative result)

- **Recursive QR is the *wrong* move on wasm — the opposite of LU.**
  ReLAPACK (ACM TOMS 10.1145/3061664 — the definitive recursive-LAPACK
  library, whose recursive LU we *did* adopt) **deliberately excludes QR,
  `ormqr`/`orgqr`, and the Hessenberg/tridiagonal/bidiagonal reductions**,
  on the grounds that recursive formulations of these orthogonal-transform
  routines cost significantly **more FLOPs** than the blocked versions.
  Combined with confirmed claim 6 (`dgeqrt3` recurses to skinny gemms) and
  the LU finding (skinny gemms lose on wasm), the verdict is clear: **do
  not port a recursive QR.** The LU playbook does not transfer. Grade:
  the ReLAPACK *exclusion* is confirmed structural fact; the stated
  *reason* (extra flops) is well-known and consistent, graded
  observed/by-hand.
- **The T-formation micro-optimizations (UT transform, alternative
  `dlarft` recurrences) are not our bottleneck.** The 2006 UT-transform
  paper (10.1145/1141885.1141886) is real and does describe a lower-flop
  `I − U T^{-1} U^T` accumulation — but `T` is r×r and its formation is
  O(r³), dwarfed by the O(mnr) panel and O(mnk) trailing update. The
  extracted inference that "an independent kernel can capture savings
  LAPACK leaves on the table" is an **overreach** for our sizes: chasing
  `T`-formation flops cannot move an n≤512 wall. Parked.

## Sourced-unverified (single-paper performance numbers; votes never ran)

Grade: **observed / sourced-unverified** — one motivated source each, no
adversarial check. Treat as directional, not settled.

- **"Blocked QR attains only 8–16% of peak Gflops on Haswell"** and
  **"Householder-QR saturates at 80–85% of the platform's GEMM"** (arXiv
  1612.04470). The second is a commonly-cited ceiling and plausible; the
  first is from a hardware-codesign paper motivating a custom PE, so its
  baseline is suspect. If even *directionally* right, the read is: QR
  cannot be routed into gemm as cleanly as LU/Cholesky — there is an
  intrinsic panel-bound residue, so an unblocked-panel strategy that is
  already gemm-competitive (ours) is not obviously beatable by blocking at
  our sizes.
- **Fusing the unblocked HT operations (norm/τ, `v^T·A`, rank-1 update)
  raises arithmetic intensity** — the paper's "MHT/DGEQR2HT" reaches ~74%
  of peak (99.3% of its PE's DGEMM) *on custom hardware*. The **numbers
  are architecture-specific and do not transfer to wasm**, but the
  **concept does**: our panel-width-1 kernel is a fusion opportunity
  (keep the reflector, its `v^T·A`, and the update in one pass over the
  trailing columns rather than three), and it is already our win path.

## What wasm-shaped QR *should be* (the positive answer)

The earlier draft of this section stopped at "not recursive" — a
constraint, not a shape. The architect (correctly) rejected that: the
question is what the kernel *should be*, given faer's default is
native-shaped and won't be wasm-optimal by luck. **Grading discipline
first — this answer is a synthesis, not a research finding, and its parts
sit at different tiers:**

- The *destination* — unblocked (panel-width-1) beats faer's blocked QR
  1.3–1.7× through n=512, and the native-shaped default losing is not
  coincidence — is **measured** (our pyodide Runs 2–4; tested/scripted).
- LAPACK dropping to unblocked below its own `NX` crossover is **proven**
  (confirmed claim 3, netlib source).
- "The compact-WY block-apply costs ~2× the flops of applying reflectors
  singly" is **textbook** (Golub & Van Loan) — asserted from knowledge,
  *not* a verified source in this pipeline (stated).
- "Wasm's 2-lane SIMD shrinks the gemm advantage enough to move the
  blocking crossover past n=512" is **my inference** — no source states
  it, the mechanism is unmeasured (stated/hypothesis).
- The concrete kernel below is a **proposal by analogy to the LU kernel**,
  unbuilt (stated). It earns a higher grade only by being built and
  beating faer's `block_size=1` path — the way `lu_factor_wk` had to beat
  `lu_factor_tuned`.

**The shape.** QR decomposes into two kernels:

1. **Panel** (`dlarfg`+`dlarf`): per column, generate the reflector
   (norm/τ/scale — one simd128 pass over the column), then update each
   trailing panel column — `dot(v, c)` then `axpy(c, v, τ·dot)`.
   Memory-bound, level-1/2.
2. **Block-apply** (`dlarfb`): accumulate a panel of reflectors into
   compact-WY `(I − V T Vᵀ)` and apply to the trailing matrix as gemms.
   Level-3, but carries the ~2× WY flop penalty.

faer's default narrows the panel and pushes everything into (2)'s gemms —
correct on wide native SIMD. The hypothesis for why it inverts on wasm:
the ~2× penalty in (2) is supposed to be paid back by gemm being much
faster than the level-2 loops, and on 2-lane f64 wasm that speedup is too
small to cover it until well past n=512. Consistent with the measured
1.3–1.7× unblocked win, but the mechanism is unverified.

So the proposed **wasm-shaped QR factorization is a fused, unblocked
Householder panel in flat/simd128** — same design philosophy as the LU
panel, reusing `axpy_simd128` plus a new `dot_simd128`, with `dot(v,c)`
and `axpy(c,v,τ·dot)` fused per trailing column. **No compact-WY, no
T-matrix, no trailing gemm** for the factorization at n ≤ 512. It is
method-identical to LU but lands on the *opposite* structural conclusion:
LU wanted recursion to *feed* gemm; QR wants to stay unblocked and *not*
feed gemm. The block-apply (2) still gets built — but as a separate
kernel whose customers are forming/applying Q and Hessenberg reduction,
not the QR factor.

The open measurement, exactly as with LU: does a hand-fused simd128
unblocked panel beat faer's already-winning `block_size=1` path (as
`lu_factor_wk` beat `lu_factor_tuned` by 10–30%)? Until measured, the
kernel proposal is graded *stated*, not *tested*.

## What this means for us — the strategic read

1. **QR is already won, and now we know *why*.** `qr_r_tuned`
   (panel width 1, `bench/src/lib.rs::run_qr_factor_tuned`) beats scipy
   1.3–1.7× at n=64–512 (`docs/benchmarks-vs-pyodide-2026-07.md` Run 2)
   because their QR is reference-Fortran-over-generic-C with no arch
   help, while ours is faer's Householder over the fast wasm gemm. The
   win is **structural and durable**, not a tuning fluke.
2. **The residual QR job is packaging, not a kernel rewrite.** Task #17
   as originally written ("blocked WY driver") is *contraindicated* by
   this research: blocking/recursion don't beat unblocked at our sizes,
   and LAPACK + ReLAPACK both concede it. What consumers need is the
   panel-width-1 params **by default** (the `faer-schur::recommended_params`
   treatment, or a `faer-wasm-defaults`-style shim), so a naive `.qr()`
   caller gets the winning path instead of faer's native-shaped default.
3. **The one kernel worth building is the block-apply `(I − V T V^T)·C`
   — for Hessenberg, not for QR.** The reusable lever is `dlarfb`-shaped
   block-application routed into gemm (`V^T·C`, then `V·(T^T·(V^T·C))`).
   It buys little on standalone QR (already won) but it is the **inner
   loop of Hessenberg reduction** (`dgehrd`/`dlahr2`, Quintana-Ortí & van
   de Geijn TOMS 2006), which feeds the eigen flank — our next and
   *largest* gap (eigvals/schur 0.3–0.4× vs scipy). Build the block-apply
   once, wasm-shaped, and it pays QR margin now and the eigen pipeline
   later.

## Ranked plan

1. **Package the panel-width-1 QR as the default** consumers get — the
   cheap, confirmed win. Mirror `faer-schur`'s `recommended_params`: a
   thin `recommended_qr_params()` (wasm → block_size 1) so `.qr()`-shaped
   calls don't fall into faer's native default. Gate: the existing
   pyodide-bench `qr_r_tuned` row, plus a correctness probe.
2. **Wasm-shaped block-apply kernel `(I − V T V^T)·C` in
   `faer-wasm-kernels`** — coarse-only (no recursion, no skinny-gemm base
   case; learn from LU), `V^T·C` and the back-multiply into faer gemm,
   the rank-1/rank-k glue in flat simd128. Correctness-gated in
   `kernels/tests` like the LU kernels. This is the durable investment.
3. **Feed (2) into a wasm-shaped Hessenberg reduction** — the actual
   eigen-flank lever; `dlahr2`-shape with the block-apply from (2). This
   is where the 2.5–3× eigvals/schur gaps live. Larger job; scope after
   (2) lands.
4. **Do NOT** build recursive QR or chase T-formation flop savings —
   refuted above.

## Regenerate / extend

Search + fetch completed (14 sources, 36 claims); the 3-vote verify phase
died on Fable-5 usage-credit exhaustion (run `wf_51e9a0ba-050`; journal in
`.../subagents/workflows/wf_51e9a0ba-050/journal.jsonl`, raw salvage in the
session scratch `tasks/wgku2l20a.output`). Re-run the deep-research
workflow with the same args once credits reset to get the independent
adversarial votes and lift the structural claims from by-hand to
cross-checked. The extracted-but-unranked tail (36−25 claims, incl. the
Hessenberg/`dlahr2` source) is in the journal for a deeper pass.
