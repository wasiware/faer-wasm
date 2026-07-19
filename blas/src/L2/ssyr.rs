//! `ssyr` — symmetric rank-1 update: A ← A + αxxᵀ, one triangle stored.
//!
//! Implementation: column-saxpy — one Level 1 `saxpy` stream per stored
//! column segment.

use super::check_mat;
use crate::L1::saxpy;

/// A ← A + αxxᵀ, A symmetric n×n at column stride `cs`, `upper` (or
/// lower) triangle stored.
pub fn ssyr(alpha: f32, n: usize, a: &mut [f32], cs: usize, upper: bool, x: &[f32]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "ssyr: x length mismatch");
	for j in 0..n {
		let t = alpha * x[j];
		if upper {
			saxpy(t, &x[..=j], &mut a[j * cs..j * cs + j + 1]);
		} else {
			saxpy(t, &x[j..], &mut a[j * cs + j..j * cs + n]);
		}
	}
}
