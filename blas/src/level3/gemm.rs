//! `gemm` — matrix multiplication: C ← αAB + βC.
//!
//! Implementation: column-axpy — gemm is exactly `gemv` per column of
//! B/C. This flat streaming shape beat faer's blocked gemm 1.07–1.33×
//! through n = 512 on the reference machines (docs/blas-ab-2026-07.md).
//! Transpose forms: not built — no consumer yet (syrk covers A·Aᵀ).

use super::check_mat;
use crate::level2::gemv;

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its own
/// column stride.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
	alpha: f64,
	m: usize,
	k: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		gemv(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}
