# src/ — the call graph

Which functions call which functions. One node per routine, covering
both types: the f64 and f32 layers have identical call structure by
construction, so `gemv → axpy` means both `dgemv → daxpy` and
`sgemv → saxpy`.

```mermaid
graph TD
  subgraph L3
    gemm
    gemm_tiled
    gemm_col4
    gemm_colaxpy
    symm_left
    symm_right
    syrk
    syr2k
    trmm_left
    trmm_right
    trsm_left
    trsm_right
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

  subgraph kernels
    axpy4
    axpy4in
    axpy_dot
    tile
  end

  lanes

  gemm --> gemm_tiled
  gemm --> gemm_col4
  gemm_tiled --> tile
  gemm_tiled --> gemv
  gemm_tiled --> axpy
  gemm_col4 --> axpy4
  gemm_col4 --> gemv
  gemm_colaxpy --> gemv
  symm_left --> symv
  symm_right --> axpy4
  symm_right --> axpy
  syrk --> axpy4
  syrk --> axpy
  syr2k --> axpy4
  syr2k --> axpy
  trmm_left --> axpy4
  trmm_left --> trmv
  trmm_left --> scal
  trsm_left --> axpy4
  trsm_left --> trsv
  trsm_left --> scal
  trmm_right --> axpy4
  trmm_right --> axpy
  trmm_right --> scal
  trsm_right --> axpy4
  trsm_right --> axpy
  trsm_right --> scal

  gemv --> axpy4in
  gemv --> axpy
  gemv_t --> dot
  ger --> axpy
  symv --> axpy_dot
  trmv --> axpy4in
  trmv --> axpy
  trsv --> axpy4in
  trsv --> axpy
  syr --> axpy
  syr2 --> axpy

  axpy --> lanes
  dot --> lanes
  scal --> lanes
  copy --> lanes
  swap --> lanes
  rot --> lanes
  nrm2 --> lanes
  asum --> lanes
  iamax --> lanes
  axpy4 --> lanes
  axpy4in --> lanes
  axpy_dot --> lanes
  tile --> lanes
```

Reading notes, kept out of the graph:

- `gemm` is a size dispatcher over `gemm_tiled` and `gemm_col4`;
  `gemm_colaxpy` is the plain reference shape, kept for bit checks.
  `tile` is gemm's private register-tile micro-kernel (it lives in
  the gemm files, not `kernels.rs`); `axpy_dot` stands for both the
  1- and 4-column fused symv kernels.
- `rotg` calls nothing (guarded scalar, no arrays); copy/swap/rot/
  nrm2/asum/iamax have no in-crate callers — consumers call them
  directly.
- Not drawn: the small type-free helpers `check_mat`,
  `{d,s}scale_y`, `{d,s}sym_at`, used across their levels.
- WHY each edge has the shape it has (fan-out/fan-in/fused/tile) is
  the crate README's taxonomy and tables; the measured consequences
  are `../bench/README.md`.

One property worth knowing: improvements flow up the arrows — when
the tuning campaign gave `dot` four accumulators, `gemv_t` got
1.3–1.7× faster untouched (two-draw runner verdict, docs step 7).
