//! `trmv` — triangular matrix × vector, in place: x ← Ax.
//!
//! Implementation: column-axpy — one Level 1 `axpy` stream per column,
//! in reference `dtrmv`'s column order (ascending for upper, descending
//! for lower) so each x[j] is consumed before its own slot is
//! overwritten. Transposed form: not built — no consumer yet (explicit
//! gap).

use super::check_mat;
use crate::level1::axpy;

/// x ← Ax, A triangular n×n at column stride `cs`. `upper` selects the
/// triangle; `unit` treats the diagonal as ones (stored values
/// ignored).
pub fn trmv(n: usize, a: &[f64], cs: usize, upper: bool, unit: bool, x: &mut [f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "trmv: x length mismatch");
	if upper {
		for j in 0..n {
			let col = &a[j * cs..j * cs + n];
			let t = x[j];
			axpy(t, &col[..j], &mut x[..j]);
			if !unit {
				x[j] = t * col[j];
			}
		}
	} else {
		for j in (0..n).rev() {
			let col = &a[j * cs..j * cs + n];
			let t = x[j];
			axpy(t, &col[j + 1..], &mut x[j + 1..]);
			if !unit {
				x[j] = t * col[j];
			}
		}
	}
}
