//! `gemv` — matrix × vector: y ← αAx + βy (and the transpose twin
//! y ← αAᵀx + βy).
//!
//! Implementation: column-axpy — one Level 1 `axpy` stream per column
//! of A (the transpose form is one `dot` reduction stream per column).
//! Evidence: docs/blas-ab-2026-07.md. The fused-FMA variant (measured
//! better in the step-1 race) lands with the relaxed-simd build
//! campaign.

use super::{check_mat, scale_y};
use crate::level1::{axpy, dot};

/// y ← αAx + βy. A is nrows×ncols at column stride `cs`;
/// x has ncols elements, y has nrows.
pub fn gemv(
	alpha: f64,
	nrows: usize,
	ncols: usize,
	a: &[f64],
	cs: usize,
	x: &[f64],
	beta: f64,
	y: &mut [f64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), ncols, "gemv: x length mismatch");
	assert_eq!(y.len(), nrows, "gemv: y length mismatch");
	scale_y(beta, y);
	for j in 0..ncols {
		axpy(alpha * x[j], &a[j * cs..j * cs + nrows], y);
	}
}

/// y ← αAᵀx + βy. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub fn gemv_t(
	alpha: f64,
	nrows: usize,
	ncols: usize,
	a: &[f64],
	cs: usize,
	x: &[f64],
	beta: f64,
	y: &mut [f64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "gemv_t: x length mismatch");
	assert_eq!(y.len(), ncols, "gemv_t: y length mismatch");
	scale_y(beta, y);
	for j in 0..ncols {
		y[j] += alpha * dot(&a[j * cs..j * cs + nrows], x);
	}
}
