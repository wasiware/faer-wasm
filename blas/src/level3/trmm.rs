//! `trmm` — triangular matrix multiply, in place:
//! B ← αAB (left) or B ← αBA (right), A triangular.
//!
//! Implementation: left side is `trmv` per column of B (plus one αscal
//! when α ≠ 1); right side is a column-axpy sweep in the column order
//! that consumes each original column before overwriting it
//! (descending for upper, ascending for lower). Fused-FMA variant
//! (measured better in the step-1 race) lands with the relaxed-simd
//! build campaign. Transpose forms: not built — no consumer yet.

use super::check_mat;
use crate::level1::{axpy, scal};
use crate::level2::trmv;

/// B ← αAB. A is m×m triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub fn trmm_left(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f64],
	bcs: usize,
) {
	check_mat(b.len(), m, n, bcs);
	for j in 0..n {
		let col = &mut b[j * bcs..j * bcs + m];
		trmv(m, a, acs, upper, unit, col);
		if alpha != 1.0 {
			scal(alpha, col);
		}
	}
}

/// B ← αBA. A is n×n triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub fn trmm_right(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f64],
	bcs: usize,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// result column j = α·Σ_{k in triangle} a[k,j]·B_orig[:,k]. The
	// column order (descending for upper, ascending for lower) touches
	// each original column before anything overwrites it. Columns j
	// and k never alias (k ≠ j always), so the two views below are
	// disjoint — raw pointers express what split_at_mut can't say
	// cleanly for arbitrary (j, k) pairs.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f64, m) };
	let mut do_col = |j: usize| {
		let diag = if unit { 1.0 } else { a[j * acs + j] };
		scal(alpha * diag, col_mut(j));
		let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
		for k in lo..hi {
			axpy(alpha * a[j * acs + k], col_ref(k), col_mut(j));
		}
	};
	if upper {
		for j in (0..n).rev() {
			do_col(j);
		}
	} else {
		for j in 0..n {
			do_col(j);
		}
	}
}
