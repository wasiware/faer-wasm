//! `zhemv` — Hermitian matrix × vector: y ← αAx + βy, A Hermitian
//! with only one triangle stored (A[j,i] = conj(A[i,j]); diagonal
//! imaginary parts are ignored, per the LAPACK storage convention).
//!
//! Implementation: fused column pass — one stream over the stored
//! strict segment serves both triangles' contributions
//! (`kernels::zaxpy_dotc`: y[i] += t·a[i] elementwise while
//! acc += conj(a[i])·x[i] reduces), the complex twin of the `dsymv`
//! fused shape at single-column width. The diagonal contributes
//! t·re(a[j,j]) (real by convention). Accumulation order is the fused
//! pass's own — zhemv is bounds-tested, not bit-locked; cross-target
//! determinism holds through the lane emulation as everywhere else.
//! The 4-column fused grouping that pushed `dsymv` to 2× is a
//! recorded tuning lever, not yet built for c64.

use super::{check_mat, zscale_y};
use crate::c64::C64;
use crate::kernels::zaxpy_dotc;

/// y ← αAx + βy, A Hermitian n×n at column stride `cs`, with the
/// `upper` (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub fn zhemv(
	alpha: C64,
	n: usize,
	a: &[C64],
	cs: usize,
	upper: bool,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "zhemv: x length mismatch");
	assert_eq!(y.len(), n, "zhemv: y length mismatch");
	zscale_y(beta, y);
	for j in 0..n {
		let cj = j * cs;
		let t = alpha * x[j];
		let d = if upper {
			unsafe { zaxpy_dotc(a.as_ptr().add(cj), t, x.as_ptr(), y.as_mut_ptr(), j) }
		} else {
			unsafe {
				zaxpy_dotc(
					a.as_ptr().add(cj + j + 1),
					t,
					x.as_ptr().add(j + 1),
					y.as_mut_ptr().add(j + 1),
					n - j - 1,
				)
			}
		};
		y[j] = y[j] + t.scale(a[cj + j].re) + alpha * d;
	}
}

/// 4-column grouped fused variant (close-out race candidate,
/// 2026-07-19): the dsymv grouping at complex lane geometry — four
/// stored columns share ONE stream over x and y
/// (`kernels::zaxpy_dotc4`), where the shipped shape streams x and y
/// once per column. Same bounds-tested contract as `zhemv` (the
/// grouping re-folds the reduction); raced against `zhemv` on the
/// reference runners before either ships as the default.
#[allow(clippy::too_many_arguments)]
pub fn zhemv_grouped(
	alpha: C64,
	n: usize,
	a: &[C64],
	cs: usize,
	upper: bool,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	use crate::kernels::zaxpy_dotc4;
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "zhemv: x length mismatch");
	assert_eq!(y.len(), n, "zhemv: y length mismatch");
	zscale_y(beta, y);
	let ap = a.as_ptr();
	let mut j = 0usize;
	while j + 4 <= n {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let cols = unsafe {
			[ap.add(j * cs), ap.add((j + 1) * cs), ap.add((j + 2) * cs), ap.add((j + 3) * cs)]
		};
		if upper {
			// common strict segment rows 0..j fused for all four; then
			// the ragged rows j..j+u and the diagonal, scalar per column
			let mut d = unsafe { zaxpy_dotc4(cols, t, x.as_ptr(), y.as_mut_ptr(), j) };
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j..j + u {
					y[i] = y[i] + t[u] * a[cj + i];
					d[u] = d[u] + a[cj + i].conj() * x[i];
				}
				y[j + u] = y[j + u] + t[u].scale(a[cj + j + u].re) + alpha * d[u];
			}
		} else {
			// common strict segment rows j+4..n; ragged rows j+u+1..j+4
			let mut d = unsafe {
				zaxpy_dotc4(
					[cols[0].add(j + 4), cols[1].add(j + 4), cols[2].add(j + 4), cols[3].add(j + 4)],
					t,
					x.as_ptr().add(j + 4),
					y.as_mut_ptr().add(j + 4),
					n - j - 4,
				)
			};
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j + u + 1..j + 4 {
					y[i] = y[i] + t[u] * a[cj + i];
					d[u] = d[u] + a[cj + i].conj() * x[i];
				}
				y[j + u] = y[j + u] + t[u].scale(a[cj + j + u].re) + alpha * d[u];
			}
		}
		j += 4;
	}
	while j < n {
		let cj = j * cs;
		let t = alpha * x[j];
		let d = if upper {
			unsafe { crate::kernels::zaxpy_dotc(a.as_ptr().add(cj), t, x.as_ptr(), y.as_mut_ptr(), j) }
		} else {
			unsafe {
				crate::kernels::zaxpy_dotc(
					a.as_ptr().add(cj + j + 1),
					t,
					x.as_ptr().add(j + 1),
					y.as_mut_ptr().add(j + 1),
					n - j - 1,
				)
			}
		};
		y[j] = y[j] + t.scale(a[cj + j].re) + alpha * d;
		j += 1;
	}
}
