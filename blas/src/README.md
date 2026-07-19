# src/ — the dependency map

One file per BLAS routine per type, netlib naming (convention:
`L1/README.md`). The layer is a strict one-way composition — every
edge points from caller to callee; no cycles, no sideways calls
within a level.

**One node = one routine family, both types.** The f64 (d-) and f32
(s-) routines have identical edge sets by construction — same files,
same calls, with the s-side running on the s-kernels and `F32x4` —
so the map treats them as one. (If a future type ever deviates
structurally, it gets its own edges here.)

```mermaid
graph TD
  subgraph L3
    gemm["gemm (size dispatch: tile / col4;\nplain colaxpy kept as bit reference)"]
    symm_left
    symm_right
    syrk["syrk / syr2k"]
    trmm_left["trmm_left / trsm_left"]
    trmm_right["trmm_right / trsm_right"]
  end

  subgraph L2
    gemv
    gemv_t
    ger["ger · syr · syr2"]
    symv
    trmv["trmv / trsv"]
  end

  subgraph L1
    axpy
    dot
    scal
    l1rest["copy · swap · rot · nrm2 · asum · iamax\n(each self-contained)  ·  rotg (scalar)"]
  end

  subgraph kernels
    axpy4["axpy4 (+FO)"]
    axpy4in["axpy4in (+FI)"]
    axpy_dot["axpy_dot(4) (FS)"]
    tile["register tile (RT, private to gemm)"]
  end

  lanes["lanes: F64x2 / F32x4"]

  gemm --> tile
  gemm --> axpy4
  gemm -->|tails| gemv
  symm_left -->|per column of B| symv
  symm_right --> axpy4
  syrk --> axpy4
  syrk -->|ragged edge + tails| axpy
  trmm_left -->|lockstep walk| axpy4
  trmm_left -->|tail columns| trmv
  trmm_right -->|out-of-group| axpy4
  trmm_right -->|in-group| axpy
  trmm_right --> scal

  gemv -->|4-col groups| axpy4in
  gemv -->|tails| axpy
  gemv_t -->|per column| dot
  ger -->|per column| axpy
  symv --> axpy_dot
  trmv -->|common segment| axpy4in
  trmv -->|tails| axpy

  axpy --> lanes
  dot --> lanes
  scal --> lanes
  l1rest --> lanes
  axpy4 --> lanes
  axpy4in --> lanes
  axpy_dot --> lanes
  tile --> lanes
```

Edge labels are the crate README's shorthand (+FO fan-out, +FI
fan-in, FS fused symv pass, RT register tile). Grouped nodes share
their edges: `syrk / syr2k` both stream via axpy4 with ragged edges
on axpy; `trmm_left / trsm_left` both do the lockstep walk (trsm's
tails go to trsv); `trmm_right / trsm_right` differ only in trsm's
reciprocal scal; `ger · syr · syr2` are all plain axpy-per-column;
symm_right's tail columns use axpy.

Not drawn: the shared helpers — `L2::check_mat` (storage validation,
type-free), `L2::{d,s}scale_y` (BLAS β=0 = hard zero-fill),
`L3::{d,s}sym_at` (stored-triangle lookup) — are leaf utilities used
across their levels.

Why composition is load-bearing and not just tidy: when the tuning
campaign gave `dot` four accumulators, `gemv_t` — a loop of dot calls
— got 1.3–1.7× faster without being touched (two-draw runner verdict,
docs step 7). Improvements flow up the arrows, in both types at once.

The composition is structural, not sacred: any edge may be replaced
by a tuned kernel when a race on the reference machines says so (the
record of every such decision: `../../docs/blas-ab-2026-07.md`).
