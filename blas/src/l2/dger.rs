//! `dger` — outer-product (rank-1) update: A ← A + αxyᵀ.
//!
//! Implementation: column-daxpy — one Level 1 `daxpy` stream per column
//! (column j gets α·y[j] times x).

use super::check_mat;
use crate::l1::daxpy;

/// A ← A + αxyᵀ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub fn dger(alpha: f64, nrows: usize, ncols: usize, a: &mut [f64], cs: usize, x: &[f64], y: &[f64]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "dger: x length mismatch");
	assert_eq!(y.len(), ncols, "dger: y length mismatch");
	for j in 0..ncols {
		daxpy(alpha * y[j], x, &mut a[j * cs..j * cs + nrows]);
	}
}
