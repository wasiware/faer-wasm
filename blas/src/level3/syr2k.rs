//! `syr2k` — symmetric rank-2k update: C ← α(ABᵀ + BAᵀ) + βC, one
//! triangle of C stored.
//!
//! Implementation: truncated column-axpy — two streams per stored
//! column segment per inner index (the second rides the cache the
//! first just warmed).

use super::check_mat;
use crate::level1::axpy;
use crate::level2::scale_y;

/// C ← α(ABᵀ + BAᵀ) + βC. A and B are n×k; C is n×n with the `upper`
/// (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn syr2k(
	alpha: f64,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	upper: bool,
) {
	check_mat(a.len(), n, k, acs);
	check_mat(b.len(), n, k, bcs);
	check_mat(c.len(), n, n, ccs);
	for j in 0..n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		scale_y(beta, seg);
		for l in 0..k {
			axpy(alpha * b[l * bcs + j], &a[l * acs + lo..l * acs + hi], seg);
			axpy(alpha * a[l * acs + j], &b[l * bcs + lo..l * bcs + hi], seg);
		}
	}
}
