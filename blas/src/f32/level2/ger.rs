//! `ger` — outer-product (rank-1) update: A ← A + αxyᵀ.
//!
//! Implementation: column-axpy — one Level 1 `axpy` stream per column
//! (column j gets α·y[j] times x).

use super::check_mat;
use crate::f32::level1::axpy;

/// A ← A + αxyᵀ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub fn ger(alpha: f32, nrows: usize, ncols: usize, a: &mut [f32], cs: usize, x: &[f32], y: &[f32]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "ger: x length mismatch");
	assert_eq!(y.len(), ncols, "ger: y length mismatch");
	for j in 0..ncols {
		axpy(alpha * y[j], x, &mut a[j * cs..j * cs + nrows]);
	}
}
