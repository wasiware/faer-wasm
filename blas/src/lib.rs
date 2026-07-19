//! faer-wasm-blas: the wasm-native BLAS layer, one file per BLAS
//! routine in netlib naming (d-prefixed f64, s-prefixed f32 — the
//! convention is documented in src/l1/README.md; the dependency map
//! in src/README.md). Both type layers are complete and tuned; unit
//! stride — callers pass contiguous column slices (strided access
//! defeats streaming and no consumer wants it). The tuned
//! multi-column loop shapes live in `kernels`; the SIMD lane types
//! with their bit-identical native emulation live in `lanes`. Plan of
//! record: README.md in the crate root.
#![no_std]

mod kernels;
mod lanes;

pub mod l1;
pub mod l2;
pub mod l3;
