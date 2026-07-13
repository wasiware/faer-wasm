# faer wasm-vs-native benchmarks — 2026-07

> **Status note (2026-07-11):** every number in this file predates the
> LIFO-rewind allocator fix (commit 6880b5a). The old leak-only bump
> allocator taxed allocation-heavy ops on the wasm side (cold pages +
> `memory.grow` inside the timing loop), so the wasm-vs-native ratios
> here are pessimistic — worst for faer's untuned default paths and the
> eigen pipelines. Within-file comparisons remain valid (both sides of
> each pair paid the same tax); cross-referencing these absolute ratios
> against post-fix runs is not. Re-measurement is queued on the ROADMAP
> watch list.

First measured answer to "how fast is faer on wasm, and which build knobs
matter". Produced by `bench/` (see §Method); regenerate any time — these
numbers are a snapshot, not a guarantee.

- Toolchain: rustc 1.94.1, node v22.22.2 (V8), pinned faer 0539947 +
  `patches/`, pulp 5eb07fd.
- Host: shared cloud container (x86-64, AVX2+FMA native baseline). Shared
  = noisy; every cell is **min of 2 full repetitions**, and ratios (wasm ÷
  native on the same box) are the transferable result, not absolute times.
- Variants: `z` = `opt-level="z"` (the size-first profile from
  `docs/wasm.md`), `o3` = `opt-level=3`; `relaxed` = built with
  `-C target-feature=+simd128,+relaxed-simd` (FMA). Native baseline is
  `opt-level=3` with default host features.

## Findings

1. **`opt-level` is the biggest knob we control: ~1.75× overall.**
   Geomean slowdown 9.8× (`z`) → 5.6× (`o3`). The size-first profile is
   the right browser-payload default, but compute-heavy consumers should
   ship `opt-level=3` (+~7% size: 446→477 KB for the bench module).
2. **Relaxed-SIMD (FMA) adds ~11% overall, up to ~25% where it counts.**
   Biggest on SVD and self-adjoint EVD at n≥128 (e.g. SVD 256:
   134.5→110.7 ms). Costs +4% size at `o3`. Worth it.
3. **The good news: large matmul runs at 1.8–1.9× native.** The
   "wasm is near native" story is real for gemm at n≥128.
4. **The finding: mid-size factorizations fall off a cliff.** Native
   scales smoothly with n; wasm does not. LU jumps from 2.8× (n=32) to
   ~20× (n=64); QR from 2.4× to ~15×; general EVD from 2.1× (n=64) to
   ~13× (n=128). The reproducible oddity that LU at n=128 (`o3-relaxed`)
   costs the same 1.14 ms as n=64 points at blocking/threshold selection,
   not raw compute: faer/gemm's cache-size heuristics and kernel
   thresholds are tuned for native targets and misfire on wasm. This is
   exactly ROADMAP Phase 3's "blocking-parameter tuning — measure, don't
   assume" item, which now has its data. Small-n matmul (~25× at n=32)
   likely shares the cause (per-call dispatch/planning overhead).
5. **Honest summary number: geomean ~5× for the best variant** over this
   grid (n=32…256, six ops) — worse than the 1.3–2× folklore, and the gap
   is concentrated in the mistuned mid-size cells rather than spread
   evenly. Fixing item 4 is the path to pulling the geomean toward the
   large-matmul ratio.

## Method

Same ops, sizes, and adaptive-iteration logic on both targets
(`bench/src/lib.rs` shared; `bench/bench.mjs` times wasm from node,
`bench/src/bin/native.rs` times native): per cell, warmup call, then
enough iterations for ~150 ms (min 3, capped at 500 and by projected
allocator leak — the wasm module uses the same leak-only bump allocator
as smoke-test and is re-instantiated per op). Each `run_*` returns a
probe double to defeat dead-code elimination.

Regenerate:

```sh
cd bench
cargo run --release --bin native native-o3 > native-o3.jsonl
for p in "release-z z" "release o3"; do set -- $p
  cargo build --lib --profile $1 --target wasm32-unknown-unknown
  cp target/wasm32-unknown-unknown/$1/bench_harness.wasm $2-plain.wasm
  RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' \
    cargo build --lib --profile $1 --target wasm32-unknown-unknown
  cp target/wasm32-unknown-unknown/$1/bench_harness.wasm $2-relaxed.wasm
done
for v in z-plain z-relaxed o3-plain o3-relaxed; do
  node bench.mjs $v.wasm wasm-$v > wasm-$v.jsonl
done
node report.mjs native-o3.jsonl wasm-*.jsonl
```

## Results (min of 2 repetitions, ns→µs/ms per operation)

### matmul

| n | native-o3 | wasm-z-plain (×) | wasm-z-relaxed (×) | wasm-o3-plain (×) | wasm-o3-relaxed (×) |
| -: | -: | -: | -: | -: | -: |
| 32 | 10.2 µs | 280.0 µs (27.58×) | 276.6 µs (27.25×) | 258.7 µs (25.48×) | 253.1 µs (24.93×) |
| 64 | 79.0 µs | 655.8 µs (8.30×) | 633.7 µs (8.02×) | 494.7 µs (6.26×) | 540.7 µs (6.84×) |
| 128 | 558.9 µs | 1.32 ms (2.35×) | 1.32 ms (2.35×) | 1.17 ms (2.09×) | 990.0 µs (1.77×) |
| 256 | 3.90 ms | 9.50 ms (2.44×) | 9.39 ms (2.41×) | 9.08 ms (2.33×) | 7.36 ms (1.89×) |

### lu_solve

| n | native-o3 | wasm-z-plain (×) | wasm-z-relaxed (×) | wasm-o3-plain (×) | wasm-o3-relaxed (×) |
| -: | -: | -: | -: | -: | -: |
| 32 | 13.1 µs | 134.4 µs (10.26×) | 112.4 µs (8.57×) | 36.2 µs (2.76×) | 40.5 µs (3.09×) |
| 64 | 57.9 µs | 1.57 ms (27.19×) | 1.56 ms (27.03×) | 1.14 ms (19.72×) | 1.14 ms (19.62×) |
| 128 | 338.5 µs | 4.90 ms (14.46×) | 4.86 ms (14.36×) | 3.70 ms (10.92×) | 1.14 ms (3.36×) |
| 256 | 2.08 ms | 16.76 ms (8.04×) | 14.70 ms (7.06×) | 12.27 ms (5.89×) | 12.21 ms (5.86×) |

### qr

| n | native-o3 | wasm-z-plain (×) | wasm-z-relaxed (×) | wasm-o3-plain (×) | wasm-o3-relaxed (×) |
| -: | -: | -: | -: | -: | -: |
| 32 | 51.2 µs | 223.9 µs (4.37×) | 189.5 µs (3.70×) | 142.0 µs (2.77×) | 121.5 µs (2.37×) |
| 64 | 229.4 µs | 4.18 ms (18.22×) | 4.00 ms (17.44×) | 3.61 ms (15.72×) | 3.49 ms (15.23×) |
| 128 | 1.05 ms | 11.02 ms (10.49×) | 10.05 ms (9.56×) | 9.97 ms (9.49×) | 8.26 ms (7.86×) |
| 256 | 5.71 ms | 54.59 ms (9.56×) | 50.04 ms (8.76×) | 50.04 ms (8.76×) | 47.62 ms (8.34×) |

### svd

| n | native-o3 | wasm-z-plain (×) | wasm-z-relaxed (×) | wasm-o3-plain (×) | wasm-o3-relaxed (×) |
| -: | -: | -: | -: | -: | -: |
| 32 | 268.6 µs | 2.19 ms (8.14×) | 1.75 ms (6.51×) | 829.7 µs (3.09×) | 780.3 µs (2.90×) |
| 64 | 1.13 ms | 19.12 ms (16.94×) | 16.23 ms (14.38×) | 10.47 ms (9.28×) | 10.13 ms (8.98×) |
| 128 | 6.21 ms | 52.23 ms (8.42×) | 39.45 ms (6.36×) | 22.12 ms (3.56×) | 19.44 ms (3.13×) |
| 256 | 33.27 ms | 336.29 ms (10.11×) | 247.40 ms (7.44×) | 134.52 ms (4.04×) | 110.66 ms (3.33×) |

### sa_evd

| n | native-o3 | wasm-z-plain (×) | wasm-z-relaxed (×) | wasm-o3-plain (×) | wasm-o3-relaxed (×) |
| -: | -: | -: | -: | -: | -: |
| 32 | 158.9 µs | 1.32 ms (8.32×) | 1.12 ms (7.05×) | 574.2 µs (3.61×) | 561.1 µs (3.53×) |
| 64 | 654.5 µs | 9.97 ms (15.23×) | 8.66 ms (13.24×) | 5.85 ms (8.94×) | 5.59 ms (8.54×) |
| 128 | 3.18 ms | 39.90 ms (12.54×) | 28.20 ms (8.87×) | 12.14 ms (3.81×) | 10.60 ms (3.33×) |
| 256 | 17.16 ms | 172.08 ms (10.03×) | 130.52 ms (7.61×) | 70.61 ms (4.12×) | 57.39 ms (3.34×) |

### gen_evd

| n | native-o3 | wasm-z-plain (×) | wasm-z-relaxed (×) | wasm-o3-plain (×) | wasm-o3-relaxed (×) |
| -: | -: | -: | -: | -: | -: |
| 32 | 395.3 µs | 2.64 ms (6.68×) | 2.43 ms (6.15×) | 966.6 µs (2.45×) | 940.7 µs (2.38×) |
| 64 | 1.76 ms | 13.43 ms (7.64×) | 11.90 ms (6.77×) | 4.00 ms (2.28×) | 3.76 ms (2.14×) |
| 128 | 9.25 ms | 165.26 ms (17.87×) | 155.70 ms (16.83×) | 120.76 ms (13.06×) | 121.04 ms (13.09×) |

### Geometric-mean slowdown vs native-o3

| target | geomean × |
| - | -: |
| wasm-z-plain | 9.76× |
| wasm-z-relaxed | 8.59× |
| wasm-o3-plain | 5.58× |
| wasm-o3-relaxed | 4.96× |

## Blocking-parameter tuning (finding 4 — resolved 2026-07-08)

Finding 4's cliffs were parameter sweeps away from disappearing. Both LU
and QR expose caller-side blocking knobs (`PartialPivLuParams`, the
Householder panel width passed to `qr_in_place`); `bench/tune.mjs` swept
them on the `o3-relaxed` wasm build (factor-only ops; min over sweep,
~100 ms/cell). Native baselines use library-default blocking.

| op | n | native (dflt) | wasm default | wasm tuned | tuned cfg | dflt → tuned | tuned vs native |
| - | -: | -: | -: | -: | - | -: | -: |
| LU | 32 | 11.0 µs | 34.7 µs | 14.9 µs | `rt=32` | 2.3× | 1.35× |
| LU | 64 | 52.0 µs | 117.5 µs | 65.2 µs | `rt=128` | 1.8× | 1.25× |
| LU | 128 | 322.5 µs | 3.10 ms | 481.5 µs | `rt=128` | 6.4× | 1.49× |
| LU | 256 | 2.13 ms | 12.29 ms | 2.66 ms | `rt=256` | 4.6× | 1.25× |
| QR | 32 | 51.6 µs | 118.2 µs | 87.1 µs | `bs=48` | 1.4× | 1.69× |
| QR | 64 | 240.7 µs | 2.26 ms | 224.3 µs | `bs=1` | 10.1× | **0.93×** |
| QR | 128 | 1.04 ms | 8.61 ms | 905.7 µs | `bs=1` | 9.5× | **0.87×** |
| QR | 256 | 5.87 ms | 45.70 ms | 5.26 ms | `bs=1` | 8.7× | **0.90×** |

(`rt` = `PartialPivLuParams::recursion_threshold`; `bs` = the Householder
block size, i.e. `Q_coeff` row count passed to `qr_in_place`.)

Diagnosis confirmed: the blocked/recursive paths are the pathology — their
small-panel gemm updates carry heavy per-call overhead on wasm. **Unblocked
kernels win through n=256**: LU with `recursion_threshold ≥ n` lands at
1.25–1.5× native, and QR with panel width 1 (classical Householder) lands
*at or below* faer's default native time. That last point suggests the
defaults are tuned for much larger matrices even natively; on wasm the
mismatch is simply catastrophic instead of mild.

Consumer guidance (also in `docs/wasm.md`): on wasm, call the low-level
factor APIs with `recursion_threshold ≥ n` (LU, measured up to 256) and
Householder block size 1 (QR, n ≥ 64). The high-level solvers
(`.partial_piv_lu()`, `.qr()`) use the untuned defaults. Beyond n=256 this
is unmeasured — re-sweep before assuming; blocked paths must win
eventually. SVD/EVD (3.3×+ untuned) route through their own
internally-parameterized bidiagonalization/tridiagonalization and likely
suffer the same class of overhead; squeezing them is recorded as a
possible follow-up, not attempted here.

## Efficiency gate (added 2026-07-09)

The one-shot numbers above are now backed by a per-push CI gate
(`bench/gate.mjs`, opt-level-3 wasm under node): op/matmul wall-time
ratios at n=128 checked against `bench/expected-ratios.json` within a ×3
band, O(n³) scaling windows (256/128 ∈ [3, 26]), and tuned-vs-default
guards asserting the §7 recipe of `docs/wasm.md` still wins (measured
2026-07-09: unblocked LU = 0.20× default, panel-1 QR = 0.24× default at
n=128). Bands are sized to the 3–10× cliff-class regressions this
harness has actually caught, so shared-runner noise can't trip them.

The harness also gained c64 ops. Recorded ratios vs f64 matmul at n=128
(node/V8, opt-3): c64 matmul 3.1×, c64 LU solve 4.1×, c64 QR 12.9× —
the first complex-arithmetic overhead numbers for this stack on wasm.

## Complexity verification and the Schur/EVD cliff (2026-07-09)

`bench/complexity.mjs` sweeps every op across n = 64…256 and fits the
empirical exponent p in t ≈ c·nᵖ (log-log least squares over n ≥ 96),
plus a step-jump detector (consecutive ratio capped at 4×(n₂/n₁)³) —
because an exponent fit cannot see a constant-factor cliff.

The jump detector immediately caught one: **faer's blocked multishift/AED
path is 2–13× slower than the unblocked `lahqr` kernel on wasm at every
size measured**, and the default `blocking_threshold = 75` walks straight
into it. Threshold sweep (node/V8, opt-3, min-of-2):

| n | real: default | real: lahqr | ratio | complex: default | complex: lahqr | ratio |
| -: | -: | -: | -: | -: | -: | -: |
| 64 | 9.5 ms | 9.3 ms | 1.0× | 12.1 ms | 13.2 ms | 0.9× |
| 96 | 202.2 ms | 15.1 ms | **13.4×** | 241.3 ms | 23.5 ms | **10.3×** |
| 128 | 224.3 ms | 32.0 ms | 7.0× | 322.1 ms | 53.1 ms | 6.1× |
| 192 | 243.2 ms | 102.2 ms | 2.4× | 448.9 ms | 200.5 ms | 2.2× |
| 256 | 549.4 ms | 260.6 ms | 2.1× | 1123.6 ms | 507.5 ms | 2.2× |
| 384 | 2489.6 ms | 726.7 ms | 3.4× | — | — | — |

Resolution: `faer-schur::{real,complex}::recommended_params()` raises the
blocking threshold on wasm32 so everything stays on `lahqr` (the
convenience APIs use it automatically; `*_in_place` still takes explicit
params). faer's own `.eigenvalues()` has no public params and keeps the
cliff — on wasm, prefer `faer-schur` for spectra at n ≥ 75.

Fitted exponents after the fix (all Θ(n³) in theory; LU sits low because
its cubic constant is small and quadratic overheads still amortize):

| op | p | op | p |
| - | -: | - | -: |
| matmul | 2.92 | matmul_c64 | 2.92 |
| lu_solve | 1.80 | lu_solve_c64 | 1.98 |
| qr | 2.24 | qr_c64 | 2.36 |
| svd | 2.86 | schur | 2.89 |
| sa_evd | 2.67 | schur_c64 | 3.13 |
| gen_evd | (cliff-distorted; exempt, see above) | | |

CI runs `complexity.mjs --gate` (n = 96/128/192, exponent window
[1.5, 3.6] + jump caps) on every push, so a future re-pin cannot
reintroduce a cliff or a complexity-class blowup silently.
