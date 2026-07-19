# bench/ — the measured benchmarks

The BLAS layer's measurement harness and its current results.
Self-contained — depends only on `faer-wasm-blas`, no faer — so it
builds in seconds and `blas/` is the complete product: code
(`../src`), correctness (`../tests`), measurement (here). The
append-only evidence trail with raw tables and run IDs is
`../../docs/blas-ab-2026-07.md`; the tables below quote its steps
7, 9, and 10.

## The scoreboard

Timing runs in the wasm runtime on the reference CI machines. Score =
**% of the machine's same-run measured ceiling**: Levels 1–2 against
the fastest same-run memory stream (GB/s), Level 3 against the
same-run register-resident arithmetic peak (GFLOP/s) — per the
re-derived success metric (distance from the ceiling; scipy/faer are
market comparisons only). Every figure is the range across **two
independent runner draws**.

### Level 1 (n = 2048, % of fastest same-run stream)

| routine | f64 | f32 | c64 |
|---|---|---|---|
| copy | 52–57% | 45–54% | 62–63% |
| swap | 93–97% | 73–76% | 100% |
| scal | 100% | 100% | 70–75% |
| dscal | | | 79–83% |
| rot | 85–86% | 66–67% | 97–99% |
| axpy | 76–77% | 59–68% | 81–83% |
| dot (c64: dotu / dotc) | 53–57% | 39–43% | 53–57% / 53–57% |
| nrm2 | 47–53% | 46% | 45% |
| asum | 44–50% | 42–45% | 41–42% |
| iamax | 17–21% | 8% | 24–25% |

The reductions (dot/nrm2/asum) read one array; their honest ceiling
is the READ path (the triad), not the fastest read-modify-write
stream — against the triad they sit at 73–100% in both types.
iamax's two-pass shape is priced conservatively (only one pass
counted); the f32 row's 8% is a recorded lever — the scalar
first-index rescan is per-element and dominates (docs step 10; a
fused single-pass candidate already lost its f64 race, step 9).
The c64 delegations earn their keep: zswap and zdrot ride the tuned
d-streams at 97–100% of the ceiling; the genuinely complex streams
(zscal/zaxpy at 70–83%, the dots at 53–57%) pay the shuffle work of
the lane product form.

### Level 2 (n = 2048, % of fastest same-run stream)

| routine | f64 | f32 | c64 |
|---|---|---|---|
| gemv | 57–64% | 59–64% | 50–56% |
| gemv_t (c64: also gemv_c) | 43–52% | 66–68% | 39–42% / 39–42% |
| ger (c64: geru / gerc) | 79–90% | 100% | 65–70% / 65–70% |
| symv (c64: hemv) | 53–54% | 48% | 19–21% |
| trmv | 60–66% | 54–59% | 47–52% |
| trsv | 59–65% | 53–57% | 47–53% |
| syr (c64: her) | 100% | 87–90% | 60–64% |
| syr2 (c64: her2) | 50–51% | 45–46% | 32–35% |

The weak c64 row is honest and expected: zhemv ships the
single-column fused pass — the 4-column grouping that pushed dsymv
to ~2× is a recorded lever, not yet built for complex — and its
19–21% (with zher2's 32–35% nearby) marks exactly where that lever
would land.

### Level 3 (n = 512, % of same-run arithmetic peak)

The f64 peak measured 13.3–15.3 GFLOP/s across draws; the f32 peak
26.6–27.7 (~1.8× — four lanes vs two, delivering almost fully). c64
scores against the f64 peak (complex arithmetic IS f64 arithmetic; a
complex multiply-add counts 8 real FLOPs).

| routine | f64 | f32 | c64 |
|---|---|---|---|
| gemm | 53–56% | 55–57% | 74–79% |
| symm_left (c64: hemm_left) | 84–86% | 79–82% | 39–41% |
| syrk (c64: herk) | 49–52% | 50–52% | 76–81% |
| syr2k (c64: her2k) | 49–51% | 48–51% | 74–78% |
| trmm_left | 46–48% | 50% | 81–86% |
| trsm_left | 46–48% | 50–52% | 78–85% |
| trmm_right | 53–55% | 54–58% | 77–85% |
| trsm_right | 53–55% | 54–57% | 77–85% |

The c64 family sits far closer to the arithmetic ceiling than the
real layers (74–86% vs 46–56%) — complex arithmetic does 4× the
FLOPs per byte moved, so the same fan-out shapes shift from
memory-limited toward compute-bound, and zgemm gets there without a
register tile. The exception inverts for the same reason:
hemm_left rides zhemv's un-grouped fused pass (the Level-2 lever
above), which is why it reads 39–41% where symm_left reads 84–86%.

Market comparison (not the metric): dgemm beats faer's blocked gemm
at every measured size, 1.25–1.8×, two draws (docs step 6; that race
lives in `../../bench/gemm-tune-ab.mjs`, which needs faer and loads
this crate's wasm alongside the faer harness's).

## Running a roofline (per level, per type)

```sh
cd blas/bench
cargo build --release --target wasm32-unknown-unknown --lib
cargo run --release --bin native l3-bits > /tmp/bits.txt
node l3-roofline.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm /tmp/bits.txt
# f32: use l3-bits-f32 and add --f32 to the script
# c64: use l3-bits-z and add --c64 to the script
```

Each script first verifies the level's determinism probes — the wasm
build must reproduce the native bit patterns exactly, and a mismatch
kills the run (expected values: `../tests/README.md`) — then times
every routine of its level and scores it against the same-run
ceiling.

## What's measured how

- **State** (`setup(n)`): plain column-major Vecs, `cs = n` — a, b
  (inputs), sym (SACRIFICIAL: triad destination and L2/L3 mutation
  target), tri (dominant diagonal, keeps solves bounded), rhs, and
  their f32 casts. Same LCG recipe and seeds as the historical
  bench-harness state. The c64 twins (az/bz/symz/triz/rhsz) are own
  LCG fills, re/im interleaved draws, same roles.
- **Ceilings**: `run_ceiling_bw` (pure single-pass triad into sym,
  3·8·n² bytes) and `run_ceiling_flops[_f32]` (register-resident
  mul+add chains, 8 accumulators). L1/L2 scores use the fastest
  same-run stream (a single triad under-caps read-modify-write mixes
  — the STREAM rationale); L3 scores use the arithmetic peak.
- **Verdict rules** (the standing method): reference machines only
  for verdicts, two draws per claim, sub-1.3× margins are
  direction-only unless unanimous; the dev container is for
  iteration. Runner draws use the temp-routing procedure in
  `../../docs/engineer-handoff-2026-07.md`. Update the scoreboard
  above only from runner draws.
