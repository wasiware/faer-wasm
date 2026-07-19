//! Level 3: matrix–matrix operations, one file per BLAS routine
//! (netlib naming — src/L1/README.md), every one structurally a loop
//! of Level 1/2 calls over the right-hand matrix's columns. Since the
//! 2026-07-19 tuning campaign every op runs its columns four at a
//! time through the shared blocked kernels (`crate::kernels`) or a
//! lockstep column walk — bit-identical to the plain composition
//! wherever the elimination order allows (the two right-side reorders
//! are documented in their files); gemm additionally size-dispatches
//! a register tile. Same matrix convention as Level 2 (column-major
//! slice + column stride, unit row stride).
//!
//! Scope (gap-lined in ../../README.md): no-transpose forms only —
//! Aᵀ variants of gemm have no consumer yet (syrk covers A·Aᵀ).

pub mod dgemm;
pub mod dsymm;
pub mod dsyr2k;
pub mod dsyrk;
pub mod dtrmm;
pub mod dtrsm;
pub mod sgemm;
pub mod ssymm;
pub mod ssyr2k;
pub mod ssyrk;
pub mod strmm;
pub mod strsm;

pub use dgemm::{dgemm, dgemm_col4, dgemm_colaxpy, dgemm_tiled};
pub use dsymm::{dsymm_left, dsymm_right};
pub use dsyr2k::dsyr2k;
pub use dsyrk::dsyrk;
pub use dtrmm::{dtrmm_left, dtrmm_right};
pub use dtrsm::{dtrsm_left, dtrsm_right};
pub use sgemm::{sgemm, sgemm_col4, sgemm_colaxpy, sgemm_tiled};
pub use ssymm::{ssymm_left, ssymm_right};
pub use ssyr2k::ssyr2k;
pub use ssyrk::ssyrk;
pub use strmm::{strmm_left, strmm_right};
pub use strsm::{strsm_left, strsm_right};

pub(crate) use super::L2::check_mat;

/// Symmetric-triangle element lookup: A[i,j] with only one triangle
/// stored (f64).
#[inline]
pub(crate) fn dsym_at(a: &[f64], cs: usize, upper: bool, i: usize, j: usize) -> f64 {
	let stored = if upper { i <= j } else { i >= j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j]
	}
}

/// f32 twin of `dsym_at`.
#[inline]
pub(crate) fn ssym_at(a: &[f32], cs: usize, upper: bool, i: usize, j: usize) -> f32 {
	let stored = if upper { i <= j } else { i >= j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j]
	}
}
