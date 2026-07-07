# smoke-test

A zero-import `no_std` consumer crate proving faer runs on
`wasm32-unknown-unknown` (with `patches/0001-fix-32bit-usize-shift.patch`
applied to a local faer checkout at the commit in
`patches/UPSTREAM-BASE.txt`).

- `run.mjs` — executes the wasm under node and hand-verifies matmul / LU /
  QR / SVD / EVD results (bit-identical to native x86-64 in the 2026-07
  verification).
- `out/*.wasm` — the prebuilt artifacts those size measurements came from
  (51 KiB matmul → ~396 KiB full suite; `_relaxed` = relaxed-SIMD FMA
  build). Committed as evidence; rebuild with
  `cargo build --target wasm32-unknown-unknown --release`.
