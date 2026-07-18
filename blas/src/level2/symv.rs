//! `symv` — symmetric matrix × vector: y ← αAx + βy, A symmetric with
//! only one triangle stored.
//!
//! Implementation: column-axpy — per stored column, one Level 1 `axpy`
//! stream (the stored triangle's contribution) plus one `dot` reduction
//! stream (the mirrored triangle's contribution). Same update order as
//! reference `dsymv`.

use super::{check_mat, scale_y};
use crate::level1::{axpy, dot};

/// y ← αAx + βy, A symmetric n×n at column stride `cs`, with the
/// `upper` (or lower) triangle stored.
pub fn symv(
	alpha: f64,
	n: usize,
	a: &[f64],
	cs: usize,
	upper: bool,
	x: &[f64],
	beta: f64,
	y: &mut [f64],
) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "symv: x length mismatch");
	assert_eq!(y.len(), n, "symv: y length mismatch");
	scale_y(beta, y);
	for j in 0..n {
		let col = &a[j * cs..j * cs + n];
		let t = alpha * x[j];
		if upper {
			// stored rows 0..=j: strict part streams both ways, then the
			// diagonal
			axpy(t, &col[..j], &mut y[..j]);
			y[j] += t * col[j] + alpha * dot(&col[..j], &x[..j]);
		} else {
			// stored rows j..n
			y[j] += t * col[j] + alpha * dot(&col[j + 1..], &x[j + 1..]);
			axpy(t, &col[j + 1..], &mut y[j + 1..]);
		}
	}
}
