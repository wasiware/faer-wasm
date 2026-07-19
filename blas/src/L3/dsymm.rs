//! `dsymm` — symmetric matrix multiply: C ← αAB + βC (left) or
//! C ← αBA + βC (right), A symmetric with one triangle stored.
//!
//! Implementation: left side is `dsymv` per column of B/C; right side
//! is a 4-column fan-out column-daxpy sweep (tuned 2026-07-19: each B
//! column streams once per four C columns — the dgemm `col4` shape —
//! with the symmetric element looked up across the stored triangle;
//! per-element rounding sequence identical to the plain sweep,
//! bit-for-bit tested).

use super::{check_mat, dsym_at};
use crate::kernels::daxpy4;
use crate::L1::daxpy;
use crate::L2::{dscale_y, dsymv};

/// C ← αAB + βC. A is m×m symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub fn dsymm_left(
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
		dsymv(alpha, m, a, acs, upper, &b[j * bcs..j * bcs + m], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// C ← αBA + βC. A is n×n symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub fn dsymm_right(
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
	let mut j = 0usize;
	while j + 4 <= n {
		let cp = c.as_mut_ptr();
		for u in 0..4 {
			let col = unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
			dscale_y(beta, col);
		}
		for k in 0..n {
			let t = [
				alpha * dsym_at(a, acs, upper, k, j),
				alpha * dsym_at(a, acs, upper, k, j + 1),
				alpha * dsym_at(a, acs, upper, k, j + 2),
				alpha * dsym_at(a, acs, upper, k, j + 3),
			];
			unsafe {
				daxpy4(
					b.as_ptr().add(k * bcs),
					t,
					cp.add(j * ccs),
					cp.add((j + 1) * ccs),
					cp.add((j + 2) * ccs),
					cp.add((j + 3) * ccs),
					m,
				);
			}
		}
		j += 4;
	}
	while j < n {
		let cj = &mut c[j * ccs..j * ccs + m];
		dscale_y(beta, cj);
		for k in 0..n {
			daxpy(alpha * dsym_at(a, acs, upper, k, j), &b[k * bcs..k * bcs + m], cj);
		}
		j += 1;
	}
}
