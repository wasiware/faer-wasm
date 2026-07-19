//! Level 2: matrix–vector operations, one file per BLAS routine
//! (netlib naming — src/l1/README.md), every one a composition of the
//! Level 1 streams over column slices. Since the 2026-07-19 tuning
//! campaign the multiply-vector family runs its columns four at a
//! time through the shared blocked kernels (`crate::kernels`); tails
//! and the rank-1/2 updates stay one Level 1 call per column. The
//! SIMD hot loops live in l1/kernels under their `target_feature`
//! annotations.
//!
//! Matrix convention (the whole crate): column-major slice with a
//! column stride — column `j` of an `nrows × ncols` matrix `a` with
//! stride `cs ≥ nrows` is `a[j*cs .. j*cs + nrows]`. Unit row stride
//! only; callers with padded faer matrices pass `col_stride()` as
//! `cs`.

pub mod dgemv;
pub mod dger;
pub mod dsymv;
pub mod dsyr;
pub mod dsyr2;
pub mod dtrmv;
pub mod dtrsv;
pub mod sgemv;
pub mod sger;
pub mod ssymv;
pub mod ssyr;
pub mod ssyr2;
pub mod strmv;
pub mod strsv;

pub use dgemv::{dgemv, dgemv_t};
pub use dger::dger;
pub use dsymv::dsymv;
pub use dsyr::dsyr;
pub use dsyr2::dsyr2;
pub use dtrmv::dtrmv;
pub use dtrsv::dtrsv;
pub use sgemv::{sgemv, sgemv_t};
pub use sger::sger;
pub use ssymv::ssymv;
pub use ssyr::ssyr;
pub use ssyr2::ssyr2;
pub use strmv::strmv;
pub use strsv::strsv;

/// Shared entry checks: the storage really contains an nrows×ncols
/// matrix at stride cs.
#[inline]
pub(crate) fn check_mat(a_len: usize, nrows: usize, ncols: usize, cs: usize) {
	assert!(cs >= nrows, "column stride below row count");
	if ncols > 0 {
		assert!(
			a_len >= cs * (ncols - 1) + nrows,
			"matrix storage too short for its dimensions"
		);
	}
}

/// y ← βy with BLAS β=0 semantics (a hard zero-fill, so stale NaN/inf
/// in y cannot leak through 0·y).
#[inline]
pub(crate) fn dscale_y(beta: f64, y: &mut [f64]) {
	if beta == 0.0 {
		y.fill(0.0);
	} else if beta != 1.0 {
		crate::l1::dscal(beta, y);
	}
}

/// f32 twin of `dscale_y`.
#[inline]
pub(crate) fn sscale_y(beta: f32, y: &mut [f32]) {
	if beta == 0.0 {
		y.fill(0.0);
	} else if beta != 1.0 {
		crate::l1::sscal(beta, y);
	}
}
