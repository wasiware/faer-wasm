# tests/ — correctness suites and the measured scoreboard

One test file per BLAS routine, mirroring `../src/` (naming:
`../src/L1/README.md`); `main.rs` per level folder is the Cargo test
target, `common.rs` holds the shared generator and the
higher-precision reference summers. Run with
`cd blas && cargo test --release` (64 tests). What each routine is
tested TO — bit-for-bit replay, error bound, residual — is defined in
the crate README's testing contract; each test file says which it
applies.

## The measured scoreboard (the performance half of the contract)

Timing runs in the wasm runtime on the reference CI machines, driven
by `../../bench/` — see "Regenerating" below. Score = **% of the
machine's same-run measured ceiling**: Levels 1–2 against the fastest
same-run memory stream (GB/s), Level 3 against the same-run
register-resident arithmetic peak (GFLOP/s). Every figure below is
the range across **two independent runner draws**; the append-only
evidence log with the raw tables and run IDs is
`../../docs/blas-ab-2026-07.md` (steps 7, 9, 10 are the draws
quoted here).

### Level 1 (n = 2048, % of fastest same-run stream)

| routine | f64 | f32 |
|---|---|---|
| copy | 52–57% | 45–54% |
| swap | 93–97% | 73–76% |
| scal | 100% | 100% |
| axpy | 76–77% | 59–68% |
| rot | 85–86% | 66–67% |
| dot | 53–57% | 39–43% |
| nrm2 | 47–53% | 46% |
| asum | 44–50% | 42–45% |
| iamax | 17–21% | 8% |

The reductions (dot/nrm2/asum) read one array; their honest ceiling
is the READ path (the triad), not the fastest read-modify-write
stream — against the triad they sit at 73–100% in both types.
iamax's two-pass shape is priced conservatively (only one pass
counted); the f32 row's 8% is a recorded lever — the scalar
first-index rescan is per-element and dominates (docs step 10; a
fused single-pass candidate already lost its f64 race, step 9).

### Level 2 (n = 2048, % of fastest same-run stream)

| routine | f64 | f32 |
|---|---|---|
| gemv | 57–64% | 59–64% |
| gemv_t | 43–52% | 66–68% |
| ger | 79–90% | 100% |
| symv | 53–54% | 48% |
| trmv | 60–66% | 54–59% |
| trsv | 59–65% | 53–57% |
| syr | 100% | 87–90% |
| syr2 | 50–51% | 45–46% |

### Level 3 (n = 512, % of same-run arithmetic peak)

The f64 peak measured 13.3–15.3 GFLOP/s across draws; the f32 peak
26.6–27.7 (~1.8× — four lanes vs two, delivering almost fully).

| routine | f64 | f32 |
|---|---|---|
| gemm | 53–56% | 55–57% |
| symm_left | 84–86% | 79–82% |
| syrk | 49–52% | 50–52% |
| syr2k | 49–51% | 48–51% |
| trmm_left | 46–48% | 50% |
| trsm_left | 46–48% | 50–52% |
| trmm_right | 53–55% | 54–58% |
| trsm_right | 53–55% | 54–57% |

Market comparison (not the metric): dgemm beats faer's blocked gemm
at every measured size, 1.25–1.8×, two draws (docs step 6).

## Regenerating

The rows come from `../bench/l{1,2,3}-roofline.mjs` (add `--f32` for
the s-routines) over the self-contained `blas-bench` wasm — build and
run instructions in `../bench/README.md`. Each run also verifies the
42 cross-target determinism probes against the native bits.
Reference-machine draws use the temp-routing procedure in
`../../docs/engineer-handoff-2026-07.md`; per the verdict rules,
container numbers are iteration-only — update this scoreboard only
from runner draws, two per claim.
