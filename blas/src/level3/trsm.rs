//! `trsm` — triangular solve, many right-hand sides, in place:
//! B ← αA⁻¹B (left) or B ← αBA⁻¹ (right).
//!
//! Implementation: left side is divide-then-column-axpy — αscal then
//! `trsv` per column of B; right side is the same elimination run
//! across columns (solved columns eliminated from later ones,
//! ascending for upper, descending for lower — reference `dtrsm`'s
//! order, including its multiply-by-reciprocal on the diagonal).
//! Fused-FMA variant lands with the relaxed-simd build campaign.
//! Transpose forms: not built — no consumer yet.

use super::check_mat;
use crate::level1::{axpy, scal};
use crate::level2::trsv;

/// B ← αA⁻¹B. A is m×m triangular, B is m×n. No singularity check —
/// a zero diagonal yields inf/NaN, as in reference BLAS.
#[allow(clippy::too_many_arguments)]
pub fn trsm_left(
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
		if alpha != 1.0 {
			scal(alpha, col);
		}
		trsv(m, a, acs, upper, unit, col);
	}
}

/// B ← αBA⁻¹. A is n×n triangular, B is m×n. Reference `dtrsm`
/// multiplies by the diagonal's reciprocal (not a division) — kept.
#[allow(clippy::too_many_arguments)]
pub fn trsm_right(
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
	// X·A = αB solved column-wise: X[:,j] = (αB[:,j] − Σ X[:,k]·a[k,j])
	// / a[j,j], where the sum runs over already-solved columns k
	// (k < j for upper, k > j for lower). Columns never alias.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f64, m) };
	let mut do_col = |j: usize, lo: usize, hi: usize| {
		if alpha != 1.0 {
			scal(alpha, col_mut(j));
		}
		for k in lo..hi {
			axpy(-a[j * acs + k], col_ref(k), col_mut(j));
		}
		if !unit {
			scal(1.0 / a[j * acs + j], col_mut(j));
		}
	};
	if upper {
		for j in 0..n {
			do_col(j, 0, j);
		}
	} else {
		for j in (0..n).rev() {
			do_col(j, j + 1, n);
		}
	}
}
