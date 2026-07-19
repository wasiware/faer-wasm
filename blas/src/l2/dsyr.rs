//! `dsyr` — symmetric rank-1 update: A ← A + αxxᵀ, one triangle stored.
//!
//! Implementation: column-daxpy — one Level 1 `daxpy` stream per stored
//! column segment.

use super::check_mat;
use crate::l1::daxpy;

/// A ← A + αxxᵀ, A symmetric n×n at column stride `cs`, `upper` (or
/// lower) triangle stored.
pub fn dsyr(alpha: f64, n: usize, a: &mut [f64], cs: usize, upper: bool, x: &[f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "dsyr: x length mismatch");
	for j in 0..n {
		let t = alpha * x[j];
		if upper {
			daxpy(t, &x[..=j], &mut a[j * cs..j * cs + j + 1]);
		} else {
			daxpy(t, &x[j..], &mut a[j * cs + j..j * cs + n]);
		}
	}
}
