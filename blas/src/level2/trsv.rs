//! `trsv` — triangular solve, one right-hand side, in place: x ← A⁻¹x.
//!
//! Implementation: divide-then-column-axpy — divide by the diagonal
//! entry, then one Level 1 `axpy` stream eliminates that unknown from
//! the remaining equations (forward for lower, backward for upper) —
//! reference `dtrsv`'s order exactly. Transposed form: not built — no
//! consumer yet (explicit gap).

use super::check_mat;
use crate::level1::axpy;

/// x ← A⁻¹x, A triangular n×n at column stride `cs`. `upper` selects
/// the triangle; `unit` treats the diagonal as ones (stored values
/// ignored). No singularity check — a zero diagonal yields inf/NaN,
/// as in reference BLAS.
pub fn trsv(n: usize, a: &[f64], cs: usize, upper: bool, unit: bool, x: &mut [f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "trsv: x length mismatch");
	if upper {
		for j in (0..n).rev() {
			let col = &a[j * cs..j * cs + n];
			if !unit {
				x[j] /= col[j];
			}
			let t = x[j];
			axpy(-t, &col[..j], &mut x[..j]);
		}
	} else {
		for j in 0..n {
			let col = &a[j * cs..j * cs + n];
			if !unit {
				x[j] /= col[j];
			}
			let t = x[j];
			axpy(-t, &col[j + 1..], &mut x[j + 1..]);
		}
	}
}
