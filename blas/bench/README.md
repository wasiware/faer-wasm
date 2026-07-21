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

| routine | f64 | f32 | c64 | c32 |
|---|---|---|---|---|
| copy | 52–57% | 45–54% | 62–63% | 52–58% |
| swap | 93–97% | 73–76% | 100% | 89–98% |
| scal | 100% | 100% | 70–75% | 77–78% |
| dscal | | | 79–83% | 93–100% |
| rot | 85–86% | 66–67% | 97–99% | 86–100% |
| axpy | 76–77% | 59–68% | 81–83% | 58–60% |
| dot (complex: dotu / dotc) | 53–57% | 39–43% | 53–57% / 53–57% | 46–54% / 46–54% |
| nrm2 | 47–53% | 46% | 45% | 45–48% |
| asum | 44–50% | 42–45% | 41–42% | 44–46% |
| iamax | 17–21% | 8% | 24–25% | 13–16% |

The reductions (dot/nrm2/asum) read one array; their honest ceiling
is the READ path (the triad), not the fastest read-modify-write
stream — against the triad they sit at 73–100% in both types.
iamax's two-pass shape is priced conservatively (only one pass
counted); the f32 row's 8% is a recorded lever — the scalar
first-index rescan is per-element and dominates (docs step 10; a
fused single-pass candidate already lost its f64 race, step 9).
The complex delegations earn their keep: swap/rot/the real-α scal
ride the tuned real streams at 86–100% of the ceiling in both
complex types; the genuinely complex streams pay the shuffle work of
the lane product form. icamax inherits isamax's known weakness (the
per-element scalar rescan — recorded lever).

### Level 2 (n = 2048, % of fastest same-run stream)

| routine | f64 | f32 | c64 | c32 |
|---|---|---|---|---|
| gemv | 57–64% | 59–64% | 50–56% | 60–63% |
| gemv_t (complex: also gemv_c) | 43–52% | 66–68% | 39–42% / 39–42% | 39–40% / 38–41% |
| ger (complex: geru / gerc) | 79–90% | 100% | 65–70% / 65–70% | 88–91% / 91–92% |
| symv (complex: hemv) | 53–54% | 48% | 16–21% ¹ | 12% |
| trmv | 60–66% | 54–59% | 47–52% | 60–65% |
| trsv | 59–65% | 53–57% | 47–53% | 54–56% |
| syr (complex: her) | 100% | 87–90% | 60–64% | 85–93% |
| syr2 (complex: her2) | 50–51% | 45–46% | 32–35% | 47–49% |

¹ zhemv ships the 4-column grouped fused pass since the close-out
race (2026-07-19): ~13% faster in ms than the single-column shape on
both draws (runs 29705606221/29705603966); the refreshed % range
spans one normal-class and one slow-class machine draw. The SAME
grouping LOST for c32 (~2% slower, both draws — container overruled
again), so chemv keeps the single-column pass; its 12% marks the
c32 hemv gap honestly — at two complexes per register the extra
blend work costs more than the saved traffic. The remaining recorded
levers here are the i*amax rescans.

### Level 3 (n = 512, % of same-run arithmetic peak)

The f64 peak measured 13.3–15.3 GFLOP/s across draws; the f32 peak
26.6–30.6 (~2× — four lanes vs two). Each complex type scores
against its own precision's real peak (complex arithmetic IS real
arithmetic; a complex multiply-add counts 8 real FLOPs).

| routine | f64 | f32 | c64 | c32 |
|---|---|---|---|---|
| gemm | 55–56% ³ | 56–57% | 88–94% ³ | 86–89% |
| symm_left (complex: hemm_left) | 81–84% | 74–77% | 59–68% ² | 53–62% |
| syrk (complex: herk) | 38–42% | 48–49% | 66–68% | 78–82% |
| syr2k (complex: her2k) | 38–42% | 47% | 67–68% | 76–81% |
| trmm_left | 46–50% | 44–46% | 72–75% | 77–78% |
| trsm_left | 47–50% | 46–47% | 72–75% | 78% |
| trmm_right | 47–48% | 55% | 69–76% | 87–89% |
| trsm_right | 47–48% | 54–55% | 69–76% | 88–89% |

² refreshed after the zhemv grouping shipped (runs
29705911344/29705909050) — hemm_left rides zhemv and moved from
39–41% to 54–61%.
³ refreshed after the packed-gemm dispatch shipped (runs
29721461249/29721465793; docs step 14; whole table re-drawn on that
pair). At n=512 the dispatch routes packed for f64 (53–56% → 55–56%
against a same-run peak; the bigger packed wins are at 1024³ and
deep-K, above this table's size) and for c64 (74–79% → 88–94% — the
packed complex register tile). f32/c32 route their pre-packed shapes
at 512 (their packed zones start higher); their row movement here —
and every non-gemm row's — is draw variance, not a code change: this
pair measured higher arithmetic ceilings than the earlier pairs
(f64 15.2/17.0 vs 13.3–15.3 GFLOP/s), which deflates unchanged rows'
percentages (e.g. f64 syrk 49–52% → 38–42% at near-identical
absolute GFLOP/s). Same-run-relative numbers move with the ceiling
draw; absolute ms/call in the run logs is the stable cross-check.

The complex families sit far closer to the arithmetic ceiling than
the real layers (~66–94% vs 38–57% outside the symm/hemm rows) —
complex arithmetic does 4× the
FLOPs per byte moved, so the same fan-out shapes shift from
memory-limited toward compute-bound. zgemm at 88–94% of peak (packed
tile) and cgemm at 86–89% (col4 fan-out, ~26–30 GFLOP/s — the
fastest absolute rows on the board) are effectively at the ceiling.
The remaining below-family rows are the hemm_lefts, riding their
hemv kernels (see the Level-2 note).

Market comparison (not the metric): the layer's gemm beats faer's
blocked gemm in every type raced, two draws each. dgemm 1.25–1.8× at
every measured size (docs step 6, `../../bench/gemm-tune-ab.mjs` —
pre-packed; packed adds 1.24–1.44× on top at 512³+); zgemm
1.49–1.71× at n=256–768 on both draws (n=128 split across draws:
1.70×/0.96× — call it a tie at the smallest size; packed adds
1.08–1.33× on top); cgemm 3.11–3.67× at every size including 128,
unanimous (docs step 13, `../../bench/cplx-gemm-ab.mjs` —
conservative against us: the blas rows do the full αAB+βC blend
where faer's row is a plain replace). The packed-vs-incumbent race
itself: `packed-gemm-ab.mjs`, runs 29720778259/29720782633, full
tables in docs step 14.

## Running a roofline (per level, per type)

```sh
cd blas/bench
cargo build --release --target wasm32-unknown-unknown --lib
cargo run --release --bin native l3-bits > /tmp/bits.txt
node l3-roofline.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm /tmp/bits.txt
# f32: use l3-bits-f32 and add --f32 to the script
# c64: use l3-bits-z and add --c64; c32: l3-bits-c and --c32
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
  LCG fills, re/im interleaved draws, same roles; the c32 twins are
  those fills cast to f32.
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
