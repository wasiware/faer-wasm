# src/ — the dependency map

One file per BLAS routine per type, netlib naming (convention:
`l1/README.md`). The whole layer is a strict one-way composition —
every arrow below points from a caller to what it calls, and there
are no cycles and no sideways calls within a level:

```
l3 ──▶ l2 ──▶ l1 ──▶ lanes (F64x2 / F32x4)
 │      │      ▲
 └──────┴──▶ kernels ──▶ lanes
```

- **`lanes.rs`** — the SIMD substrate: `F64x2` (two f64 lanes) and
  `F32x4` (four f32 lanes), wasm `v128` on wasm32 and a bit-identical
  elementwise emulation off-wasm. Every fold order is fixed, so
  native and wasm produce the same bits by construction. Depends on
  nothing.
- **`kernels.rs`** — the tuned multi-column stream specializations
  (d-/s-prefixed pairs): `daxpy4`/`saxpy4` (fan-out),
  `daxpy4in`/`saxpy4in` (fan-in), `daxpy_dot`(`4`)/`saxpy_dot`(`4`)
  (fused symv pass). Depend only on `lanes`.
- **`l1/`** — each routine is a self-contained stream over `lanes`;
  no l1 routine calls another l1 routine.
- **`l2/`** — compositions of l1 calls over column slices, except
  where a tuned kernel replaces the composition's inner loop.
- **`l3/`** — compositions of l2/l1 calls per column of the
  right-hand matrix, again with kernels replacing inner loops where
  the tuning campaign measured a win.

## Who calls what (per f64 routine; the s-twin mirrors it exactly)

| routine | calls |
|---|---|
| `dgemv` | `daxpy4in` (4-column groups), `daxpy` (tail) |
| `dgemv_t` | `ddot` per column |
| `dger` | `daxpy` per column |
| `dsymv` | `daxpy_dot4` / `daxpy_dot` (fused, both triangles per pass) |
| `dtrmv`, `dtrsv` | `daxpy4in` (common segment), `daxpy` (tail), scalar band |
| `dsyr`, `dsyr2` | `daxpy` per stored column segment |
| `dgemm` | dispatch → `dgemm_tiled` (register tile + `dgemv` tails) or `dgemm_col4` (`daxpy4` + `dgemv` tails); `dgemm_colaxpy` = `dgemv` per column (kept as the bit reference) |
| `dsymm_left` | `dsymv` per column of B |
| `dsymm_right` | `daxpy4` groups + `daxpy` tail columns |
| `dsyrk`, `dsyr2k` | `daxpy4` over the common triangle segment, scalar ragged edge, `daxpy` tail columns |
| `dtrmm_left`, `dtrsm_left` | lockstep column walk over `daxpy4`; `dtrmv`/`dtrsv` tail columns |
| `dtrmm_right`, `dtrsm_right` | `daxpy4` for out-of-group source columns, `daxpy` in-group, `dscal` |

Shared helpers: `l2::check_mat` (storage validation, type-free),
`l2::dscale_y`/`sscale_y` (BLAS β semantics: β=0 is a hard
zero-fill), `l3::dsym_at`/`ssym_at` (stored-triangle lookup).

Why composition is load-bearing and not just tidy: when the tuning
campaign gave `ddot` four accumulators, `dgemv_t` — a loop of ddot
calls — got 1.3–1.7× faster without being touched (two-draw runner
verdict, docs step 7). Improvements flow up the arrows.

The composition is structural, not sacred: any inner loop may be
replaced by a tuned kernel when a race on the reference machines
says so (the record of every such decision:
`../../docs/blas-ab-2026-07.md`).
