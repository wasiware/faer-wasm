# tests/ — correctness suites and the bit-identical results

One test file per BLAS routine, mirroring `../src/` (naming:
`../src/L1/README.md`); `main.rs` per level folder is the Cargo test
target, `common.rs` holds the shared generator and the
higher-precision reference summers. Run with
`cd blas && cargo test --release` — **64 tests, all green**.
Benchmarks live in `../bench/README.md`; this page is the
correctness half of the contract.

## What each routine is tested to

The standard is the strongest the routine's math allows (the full
contract rationale is in the crate README). Both types are tested
identically; the f32 bounds use an f64-accumulated reference
(products formed in f32, as the implementation forms them).

| routine(s) | standard |
|---|---|
| copy, swap, scal, axpy, rot | **bit-for-bit** vs the scalar definition |
| dot, asum | error-bounded vs compensated/f64 reference |
| nrm2 | error-bounded + over/underflow guard cases at each type's range (1e±300 / 1e±30) |
| iamax | **exact index**, incl. BLAS first-occurrence tie-breaking |
| rotg | defining identities + reference edge cases + subnormal smoke |
| gemv, ger, syr, syr2, trmv, trsv | **bit-for-bit same-order scalar replay** + independent bounds (trsv also residual-verified) |
| symv | error-bounded (the fused pass legitimately re-folds its reduction) |
| gemm, syrk, syr2k | **bit-for-bit replay** + bounds; gemm's three shapes (colaxpy / tile / col4) cross-checked bit-identical, so the size dispatch is invisible |
| symm | left: bounds (rides symv); right: **bit-for-bit replay** |
| trmm, trsm | **bit-for-bit replay** both sides + bounds/residuals — the two elimination-order reorders (trmm_right-upper, trsm_right-lower) are documented in their files and the replays mirror them; the other triangles replay the PLAIN order, proving the blocked shapes bit-identical to it |

## Cross-target bit-identity — the results

The standing guarantee: our code produces **identical bits on native
x86-64 and wasm**, by construction (`../src/lanes.rs` emulates the
SIMD lane structure elementwise off-wasm, and every reduction folds
its lanes in a fixed order). It is verified by 42 determinism probes
— fixed-LCG inputs at odd sizes so every tail path runs, folded to
one f64 — checked on every roofline run and on every reference-runner
draw to date, **all green on every check**.

The expected patterns (a probe value changes ONLY when an
accumulation order is changed deliberately, like the symv fold — any
other change is a bug, not noise):

| probe | f64 | f32 |
|---|---|---|
| L1 dot | `bff59b5a2e61f7e1` | `bff59b59e0000000` |
| L1 asum | `407fb452fdf31648` | `407fb452c0000000` |
| L1 nrm2 | `4032713910a7520b` | `4032713900000000` |
| L1 iamax | `404e800000000000` | `404e800000000000` |
| L2 gemv | `40e795d727449339` | `40e795d700000000` |
| L2 gemv_t | `40e7935bc10f043d` | `40e7935bc0000000` |
| L2 ger | `40887f34f83468a4` | `40887f3500000000` |
| L2 symv | `40e794809ffc051f` | `40e79480a0000000` |
| L2 trmv | `40f0c94074d9a36b` | `40f0c94080000000` |
| L2 trsv | `40841856de1ef82e` | `4084185700000000` |
| L2 syr | `40880d020a903f10` | `40880d0220000000` |
| L2 syr2 | `408811beb2690464` | `408811bec0000000` |
| L3 gemm | `40f891007480cf34` | `40f89100a0000000` |
| L3 symm_left | `4107979134f4a1b1` | `4107979100000000` |
| L3 syrk | `411de8614adbe916` | `411de86140000000` |
| L3 syr2k | `40f54985b47f97c9` | `40f54985a0000000` |
| L3 trmm_left | `410795b7692150e4` | `410795b7a0000000` |
| L3 trsm_left | `40268695800105e7` | `4026869540000000` |
| L3 trmm_right | `41079785fcc3c6a7` | `4107978600000000` |
| L3 trsm_right | `402684638bea76a3` | `4026846380000000` |
| L3 symm_right | `4107947163c33400` | `4107947140000000` |

Regenerate/verify: `cd ../bench && cargo run --release --bin native
l{1,2,3}-bits[-f32]` for the native side; the roofline scripts
compare the wasm build against a bits file and abort on any mismatch.
These values have been continuous through the tuning campaign's
bit-preserving levers and the 2026-07-19 restructures (verified at
each move).

Known limits of the probes, recorded honestly: the L2/L3 probe folds
are dominated by their solve-safe diagonals (~5e4), so they are weak
detectors of reduction-order changes below one ULP of the total
(docs step 8) — the per-routine replay tests above are the strong
order locks; the probes' job is cross-target identity, which they
gate hard.
