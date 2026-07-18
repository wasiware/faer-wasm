//! `symm` — symmetric matrix multiply: C ← αAB + βC (left) or
//! C ← αBA + βC (right), A symmetric with one triangle stored.
//!
//! Implementation: left side is `symv` per column of B/C; right side
//! is a column-axpy sweep with the symmetric element looked up across
//! the stored triangle.

use super::{check_mat, sym_at};
use crate::level1::axpy;
use crate::level2::{scale_y, symv};

/// C ← αAB + βC. A is m×m symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub fn symm_left(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(b.len(), m, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		symv(alpha, m, a, acs, upper, &b[j * bcs..j * bcs + m], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// C ← αBA + βC. A is n×n symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub fn symm_right(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		let cj = &mut c[j * ccs..j * ccs + m];
		scale_y(beta, cj);
		for k in 0..n {
			axpy(alpha * sym_at(a, acs, upper, k, j), &b[k * bcs..k * bcs + m], cj);
		}
	}
}
