# Instructions for Claude

This repo is the staging ground for making **faer** (pure-Rust linear
algebra) a first-class `wasm32` citizen. It exists to serve a consumer
(Ruju, a Rust reimplementation of Julia for WebAssembly), but everything
here is scoped **language-agnostically** — work only on things any
wasm-targeting consumer of faer would want.

**Prime directive — thin fork.** Every change's destination is an upstream
PR (faer's home is Codeberg; GitHub is a mirror). The repo's success
criterion is carrying *nothing*: when a phase's PRs are merged upstream,
delete the corresponding material here. Never let a long-lived divergence
accumulate.

Start by reading `README.md`, then `ROADMAP.md` (the phased plan — work the
lowest unfinished phase), then `docs/research-faer-wasm-2026-07.md` (the
empirical evidence behind the plan: measured sizes, pulp simd128 status,
the LinearAlgebra coverage matrix).

## Working setup

- The upstream clones live at `faer-rs/` and `pulp/` in the repo root
  (gitignored — never commit them). Pin them to the commits in
  `patches/UPSTREAM-BASE.txt` and apply `patches/*.patch` to `faer-rs/`.
- Install the target if missing: `rustup target add wasm32-unknown-unknown`.

## Verification (the gate for any change)

```sh
cd smoke-test
cargo build --target wasm32-unknown-unknown --release
node run.mjs   # matmul / LU / QR / SVD / EVD, hand-verified values
```

Results were bit-identical between native x86-64 and wasm in the 2026-07
verification — treat any cross-target difference as a bug, not noise. When
touching kernels or features, re-measure `.wasm` sizes and compare against
the table in `docs/` (51 KiB matmul → ~396 KiB full suite).

## Upstream contributions

Prepare everything in-repo: the branch on the local `faer-rs/` clone, the
commit with a regression test, and the PR description as a markdown file
under `upstream/`. **The human files the PR on Codeberg** — sessions here
have GitHub access only. Reference faer-rs#222 (existing wasm demand) when
relevant.

## Commits

Author every commit as `andy-emerson <emerson.andrew@gmail.com>`, credit
yourself with a `Co-Authored-By` trailer, develop on `main`, and verify
local and remote are in sync after pushing.
