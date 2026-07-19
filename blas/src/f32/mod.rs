//! The f32 layer — the tuned f64 layer cloned one-for-one (Andy,
//! 2026-07-19: "f32 layer please, same treatment as f64"). Same
//! files, same loop shapes, same testing contract; the lane type is
//! four f32 lanes per register (`lanes::F32x4`) instead of two f64
//! lanes, so every element-stride loop runs twice the elements per
//! iteration and gemm's register tile covers 8 rows instead of 4.
//! Consumer path: `faer_wasm_blas::f32::level{1,2,3}::*` (module named
//! after the type, like `core::f32`).

mod kernels;
mod lanes;

pub mod level1;
pub mod level2;
pub mod level3;
