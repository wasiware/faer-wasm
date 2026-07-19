//! `ssyr2` — symmetric rank-2 update: A ← A + α(xyᵀ + yxᵀ), one
//! triangle stored.
//!
//! Implementation: column-saxpy — two Level 1 `saxpy` streams per stored
//! column segment (the second rides the cache the first just warmed).

use super::check_mat;
use crate::l1::saxpy;

/// A ← A + α(xyᵀ + yxᵀ), A symmetric n×n at column stride `cs`,
/// `upper` (or lower) triangle stored.
pub fn ssyr2(alpha: f32, n: usize, a: &mut [f32], cs: usize, upper: bool, x: &[f32], y: &[f32]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "ssyr2: x length mismatch");
	assert_eq!(y.len(), n, "ssyr2: y length mismatch");
	for j in 0..n {
		let (ty, tx) = (alpha * y[j], alpha * x[j]);
		if upper {
			let col = &mut a[j * cs..j * cs + j + 1];
			saxpy(ty, &x[..=j], col);
			saxpy(tx, &y[..=j], col);
		} else {
			let col = &mut a[j * cs + j..j * cs + n];
			saxpy(ty, &x[j..], col);
			saxpy(tx, &y[j..], col);
		}
	}
}
