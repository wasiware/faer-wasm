# src/ — the call graph

Which routines call which routines — the crate README's 23 routines,
nothing below that level; 24 nodes because gemv's transpose twin is
drawn separately (it depends on dot where gemv depends on axpy —
they share one file and one count everywhere else). Covers both
types (the f64 and f32 layers have identical structure).

```mermaid
graph TD
  subgraph L3
    gemm
    symm
    syrk
    syr2k
    trmm
    trsm
  end

  subgraph L2
    gemv
    gemv_t
    ger
    symv
    trmv
    trsv
    syr
    syr2
  end

  subgraph L1
    axpy
    dot
    scal
    copy
    swap
    rot
    nrm2
    asum
    iamax
    rotg
  end

  gemm --> gemv
  symm --> symv
  symm --> axpy
  syrk --> axpy
  syr2k --> axpy
  trmm --> trmv
  trmm --> axpy
  trmm --> scal
  trsm --> trsv
  trsm --> axpy
  trsm --> scal

  gemv --> axpy
  gemv_t --> dot
  ger --> axpy
  trmv --> axpy
  trsv --> axpy
  syr --> axpy
  syr2 --> axpy
```

Notes:

- Below the routines sits shared plumbing that is deliberately not in
  the graph: the private SIMD kernels (`kernels.rs` — the blocked hot
  loops several routines share) and the lane types (`lanes.rs`).
  gemm's three internal loop shapes and its size dispatcher likewise
  live inside the gemm files.
- `symv` has no outgoing arrows: its fused kernel replaced the
  axpy+dot composition it used to be.
- `rotg` and copy/swap/rot/nrm2/asum/iamax are leaves — nothing in
  the crate calls them; consumers do.
