//! `dsyr2` — symmetric rank-2 update: A ← A + α(xyᵀ + yxᵀ), one
//! triangle stored.
//!
//! Implementation: column-daxpy — two Level 1 `daxpy` streams per stored
//! column segment (the second rides the cache the first just warmed).

use super::check_mat;
use crate::L1::daxpy;

/// A ← A + α(xyᵀ + yxᵀ), A symmetric n×n at column stride `cs`,
/// `upper` (or lower) triangle stored.
pub fn dsyr2(alpha: f64, n: usize, a: &mut [f64], cs: usize, upper: bool, x: &[f64], y: &[f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "dsyr2: x length mismatch");
	assert_eq!(y.len(), n, "dsyr2: y length mismatch");
	for j in 0..n {
		let (ty, tx) = (alpha * y[j], alpha * x[j]);
		if upper {
			let col = &mut a[j * cs..j * cs + j + 1];
			daxpy(ty, &x[..=j], col);
			daxpy(tx, &y[..=j], col);
		} else {
			let col = &mut a[j * cs + j..j * cs + n];
			daxpy(ty, &x[j..], col);
			daxpy(tx, &y[j..], col);
		}
	}
}
