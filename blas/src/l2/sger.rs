//! `sger` — outer-product (rank-1) update: A ← A + αxyᵀ.
//!
//! Implementation: column-saxpy — one Level 1 `saxpy` stream per column
//! (column j gets α·y[j] times x).

use super::check_mat;
use crate::l1::saxpy;

/// A ← A + αxyᵀ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub fn sger(alpha: f32, nrows: usize, ncols: usize, a: &mut [f32], cs: usize, x: &[f32], y: &[f32]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "sger: x length mismatch");
	assert_eq!(y.len(), ncols, "sger: y length mismatch");
	for j in 0..ncols {
		saxpy(alpha * y[j], x, &mut a[j * cs..j * cs + nrows]);
	}
}
