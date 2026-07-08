# smoke-test

A zero-import `no_std` consumer crate proving faer runs on
`wasm32-unknown-unknown` (with `patches/*.patch` applied to a local faer
checkout at the commit in `patches/UPSTREAM-BASE.txt`).

Feature flags stage what gets built: no features = matmul only, `lu`
adds the LU solve, `full` adds QR / SVD / both EVDs.

- `check.mjs` — **the gate.** Runs the wasm under node and compares the
  probe values *exactly* (bit-identical to native x86-64), plus a size
  check against `size-budgets.json`. Usage:
  `node check.mjs [wasm-path] [matmul|lu|full|full-relaxed]`.
- `run.mjs` — human-facing quick look: prints exports and values without
  judging them.

CI (`.github/workflows/wasm-gate.yml`) builds all four variants on every
push and gates on `check.mjs`. Current sizes and budgets live in
`docs/wasm.md` §3.
