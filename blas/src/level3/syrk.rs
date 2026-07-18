//! `syrk` — symmetric rank-k (Gram-matrix) update:
//! C ← αAAᵀ + βC, one triangle of C stored.
//!
//! Implementation: truncated column-axpy — stored column j of C
//! accumulates α·A[j,l]·A[range,l] over the inner dimension, streaming
//! only the stored segment (the raced triangular-aware loop that read
//! at parity-or-better with faer, skipping half the writes). The
//! plain (non-FMA) variant is the measured choice: fusing HARMED syrk
//! on all three step-1 draws.

use super::check_mat;
use crate::level1::axpy;
use crate::level2::scale_y;

/// C ← αAAᵀ + βC. A is n×k; C is n×n with the `upper` (or lower)
/// triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn syrk(
	alpha: f64,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	upper: bool,
) {
	check_mat(a.len(), n, k, acs);
	check_mat(c.len(), n, n, ccs);
	for j in 0..n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		scale_y(beta, seg);
		for l in 0..k {
			axpy(alpha * a[l * acs + j], &a[l * acs + lo..l * acs + hi], seg);
		}
	}
}
