//! `dgemv` — matrix × vector: y ← αAx + βy (and the transpose twin
//! y ← αAᵀx + βy).
//!
//! Implementation: 4-column fan-in column-daxpy (tuned 2026-07-19) —
//! groups of four A columns stream through one pass over y, so y's
//! read-modify-write traffic drops 4× versus one `daxpy` per column;
//! the rounding sequence per element is identical to the plain
//! column-daxpy loop (bit-for-bit tested). Tail columns fall back to
//! Level 1 `daxpy`. The transpose form is one `ddot` reduction stream
//! per column. Evidence: docs/blas-ab-2026-07.md. The fused-FMA
//! variant (measured better in the step-1 race) is DEFERRED at the
//! f64 campaign close (relaxed-madd rounding is implementation-
//! dependent — architect decision, recorded in ROADMAP).

use super::{check_mat, dscale_y};
use crate::kernels::daxpy4in;
use crate::L1::{daxpy, ddot};

/// y ← αAx + βy. A is nrows×ncols at column stride `cs`;
/// x has ncols elements, y has nrows.
#[allow(clippy::too_many_arguments)]
pub fn dgemv(
	alpha: f64,
	nrows: usize,
	ncols: usize,
	a: &[f64],
	cs: usize,
	x: &[f64],
	beta: f64,
	y: &mut [f64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), ncols, "dgemv: x length mismatch");
	assert_eq!(y.len(), nrows, "dgemv: y length mismatch");
	dscale_y(beta, y);
	let mut j = 0usize;
	while j + 4 <= ncols {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let ap = a.as_ptr();
		unsafe {
			daxpy4in(
				ap.add(j * cs),
				ap.add((j + 1) * cs),
				ap.add((j + 2) * cs),
				ap.add((j + 3) * cs),
				t,
				y.as_mut_ptr(),
				nrows,
			);
		}
		j += 4;
	}
	while j < ncols {
		daxpy(alpha * x[j], &a[j * cs..j * cs + nrows], y);
		j += 1;
	}
}

/// y ← αAᵀx + βy. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub fn dgemv_t(
	alpha: f64,
	nrows: usize,
	ncols: usize,
	a: &[f64],
	cs: usize,
	x: &[f64],
	beta: f64,
	y: &mut [f64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "dgemv_t: x length mismatch");
	assert_eq!(y.len(), ncols, "dgemv_t: y length mismatch");
	dscale_y(beta, y);
	for j in 0..ncols {
		y[j] += alpha * ddot(&a[j * cs..j * cs + nrows], x);
	}
}
