# bench/ — the layer measures itself

The BLAS layer's measurement harness: roofline rows, cross-target
determinism probes, and machine-ceiling probes for both number types.
Self-contained — depends only on `faer-wasm-blas`, no faer — so it
builds in seconds and the `blas/` folder is the complete product:
code (`../src`), correctness (`../tests`), measurement (here). The
current two-draw results live in `../tests/README.md` (the
scoreboard); the append-only evidence trail is
`../../docs/blas-ab-2026-07.md`.

## Running a roofline (per level, per type)

```sh
cd blas/bench
cargo build --release --target wasm32-unknown-unknown --lib
cargo run --release --bin native l3-bits > /tmp/bits.txt
node l3-roofline.mjs target/wasm32-unknown-unknown/release/blas_bench.wasm /tmp/bits.txt
# f32: use l3-bits-f32 and add --f32 to the script
```

Each script first verifies the determinism probes — the wasm build
must reproduce the native bit patterns exactly (the lane-emulation
guarantee), and a mismatch kills the run — then times every routine
of its level over the persistent state and scores achieved GB/s
(L1/L2) or GFLOP/s (L3) against the same-run measured ceiling.

## What's measured how

- **State** (`setup(n)`): plain column-major Vecs, `cs = n` — a, b
  (inputs), sym (SACRIFICIAL: triad destination and L2/L3 mutation
  target), tri (dominant diagonal, keeps solves bounded), rhs, and
  their f32 casts. Same LCG recipe and seeds as the historical
  bench-harness state.
- **Determinism probes** (`run_l{1,2,3}_probe[_f32]`, 42 total):
  self-contained fixed-LCG inputs at odd sizes (tails everywhere),
  folded to one f64. Bit-compared native ↔ wasm on every script run;
  the values are continuous with the pre-move harness (verified at
  the 2026-07-19 move against the step-7/9/10 runner logs).
- **Ceilings**: `run_ceiling_bw` (pure single-pass triad into sym,
  3·8·n² bytes) and `run_ceiling_flops[_f32]` (register-resident
  mul+add chains, 8 accumulators). L1/L2 scores use the fastest
  same-run stream (a single triad under-caps read-modify-write mixes
  — the STREAM rationale); L3 scores use the arithmetic peak.
- **Verdict rules** (the standing method): reference machines only
  for verdicts, two draws per claim, sub-1.3× margins are
  direction-only unless unanimous; the dev container is for
  iteration.

## What stays in ../../bench

Everything that needs faer: the LAPACK-layer benches, the
pyodide/scipy head-to-heads, the legacy A/B harnesses
(`blas-ab.mjs`, `l1-ab.mjs`, `ceilings.mjs`), and the gemm market
race `gemm-tune-ab.mjs` — that one loads BOTH wasm modules (the faer
reference from bench-harness, the blas rows from this crate) for a
same-process interleaved race.
