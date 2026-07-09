//! Blocked LU with partial pivoting, `dgetrf`-shaped, for f64 on wasm.
//!
//! Structure per block step `j` (block width `nb`):
//!   1. **panel**: factor columns `j..j+kb` over rows `j..m` with flat
//!      scalar loops (pivot search, row swap, scale, rank-1 updates) — the
//!      O(n²·nb) part, written in the code shape wasm engines compile well;
//!   2. **pivots**: apply the panel's row swaps to the columns outside it;
//!   3. **trsm**: `A12 ← L11⁻¹ A12` (unit lower, forward substitution),
//!      lean loops over contiguous columns;
//!   4. **gemm**: `A22 ← A22 − A21·A12` via `faer::linalg::matmul` — the
//!      O(n³) bulk rides the fast microkernels.
//!
//! Pivot indices follow LAPACK `ipiv` semantics: at step `k`, rows `k` and
//! `piv[k]` were swapped (`piv[k] ≥ k`).

use faer::linalg::matmul::matmul;
use faer::prelude::*;
use faer::{Accum, MatMut};

/// `dst[i] -= src[i] * alpha` — the panel/trsm/solve workhorse. On wasm this
/// is an explicit simd128 kernel (2 lanes × 2-unrolled; `v128_load`/`store`
/// are alignment-free by spec); elsewhere a plain scalar loop. This is the
/// "shaping": the same axpy faer expresses through generic SIMD dispatch,
/// written the way the target actually executes it.
#[inline(always)]
unsafe fn axpy(dst: *mut f64, src: *const f64, alpha: f64, len: usize) {
	#[cfg(target_arch = "wasm32")]
	{
		axpy_simd128(dst, src, alpha, len);
	}
	#[cfg(not(target_arch = "wasm32"))]
	{
		for i in 0..len {
			*dst.add(i) -= *src.add(i) * alpha;
		}
	}
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn axpy_simd128(dst: *mut f64, src: *const f64, alpha: f64, len: usize) {
	use core::arch::wasm32::*;
	let va = f64x2_splat(alpha);
	let mut i = 0usize;
	while i + 4 <= len {
		let d0 = v128_load(dst.add(i) as *const v128);
		let s0 = v128_load(src.add(i) as *const v128);
		let d1 = v128_load(dst.add(i + 2) as *const v128);
		let s1 = v128_load(src.add(i + 2) as *const v128);
		v128_store(dst.add(i) as *mut v128, f64x2_sub(d0, f64x2_mul(s0, va)));
		v128_store(dst.add(i + 2) as *mut v128, f64x2_sub(d1, f64x2_mul(s1, va)));
		i += 4;
	}
	while i < len {
		*dst.add(i) -= *src.add(i) * alpha;
		i += 1;
	}
}

/// Block width. Swept on wasm 2026-07-09 (see docs/benchmarks-2026-07.md):
/// panels narrower than ~24 waste gemm efficiency, wider ones spend too
/// long in the O(n²·nb) panel at these sizes.
pub const RECOMMENDED_BLOCK_SIZE: usize = 64;

/// Factors a square `A` in place into `P·A = L·U` (`L` unit lower, `U`
/// upper, both stored in `a`), recording LAPACK-style pivots in `piv`.
/// `nb = 0` uses [`RECOMMENDED_BLOCK_SIZE`].
pub fn lu_factor_in_place(mut a: MatMut<'_, f64>, piv: &mut [usize], nb: usize) {
	let n = a.nrows();
	assert!(a.ncols() == n, "square only for now");
	assert!(piv.len() >= n);
	assert!(a.row_stride() == 1, "column-major with unit row stride required");
	// pure-panel mode through n=128: the lean panel alone matches scipy at
	// small n, and skinny-k gemm calls cost more than they save there
	let nb = if nb == 0 {
		if n <= 128 { n.max(1) } else { RECOMMENDED_BLOCK_SIZE }
	} else {
		nb
	};
	let cs = a.col_stride() as usize;
	let base = a.as_ptr_mut();

	let mut j = 0usize;
	while j < n {
		let kb = Ord::min(nb, n - j);

		// 1. panel over rows j..n, columns j..j+kb
		unsafe {
			for k in 0..kb {
				let jc = j + k;
				let col = base.add(jc * cs);
				// pivot search below (and including) the diagonal
				let mut p = jc;
				let mut mx = (*col.add(jc)).abs();
				let mut i = jc + 1;
				while i < n {
					let v = (*col.add(i)).abs();
					if v > mx {
						mx = v;
						p = i;
					}
					i += 1;
				}
				piv[jc] = p;
				// swap rows jc <-> p across the panel columns
				if p != jc {
					let mut c = j;
					while c < j + kb {
						let pc = base.add(c * cs);
						let t = *pc.add(jc);
						*pc.add(jc) = *pc.add(p);
						*pc.add(p) = t;
						c += 1;
					}
				}
				let d = *col.add(jc);
				if d != 0.0 {
					// scale the multipliers
					let inv = 1.0 / d;
					let mut i = jc + 1;
					while i < n {
						*col.add(i) *= inv;
						i += 1;
					}
				}
				// rank-1 update of the remaining panel columns
				let mut l = k + 1;
				while l < kb {
					let colr = base.add((j + l) * cs);
					let alk = *colr.add(jc);
					if alk != 0.0 {
						axpy(colr.add(jc + 1), col.add(jc + 1), alk, n - jc - 1);
					}
					l += 1;
				}
			}

			// 2. apply the panel's row swaps to columns outside the panel
			for k in 0..kb {
				let jc = j + k;
				let p = piv[jc];
				if p != jc {
					let mut c = 0usize;
					while c < n {
						if c == j {
							c = j + kb; // skip the panel: already swapped
							continue;
						}
						let pc = base.add(c * cs);
						let t = *pc.add(jc);
						*pc.add(jc) = *pc.add(p);
						*pc.add(p) = t;
						c += 1;
					}
				}
			}

			// 3. trsm: A12 <- L11^-1 A12 (unit lower forward substitution),
			// one contiguous column of A12 at a time
			if j + kb < n {
				let mut c = j + kb;
				while c < n {
					let xc = base.add(c * cs);
					for k in 0..kb {
						let xk = *xc.add(j + k);
						if xk != 0.0 {
							let lc = base.add((j + k) * cs);
							axpy(xc.add(j + k + 1), lc.add(j + k + 1), xk, kb - k - 1);
						}
					}
					c += 1;
				}
			}
		}

		// 4. gemm: A22 -= A21 * A12 — the O(n³) bulk on faer's microkernels
		if j + kb < n {
			let (_, a12_full, a21_full, a22) = a.rb_mut().split_at_mut(j + kb, j + kb);
			let a12 = a12_full.rb().subrows(j, kb);
			let a21 = a21_full.rb().subcols(j, kb);
			matmul(a22, Accum::Add, a21, a12, -1.0, Par::Seq);
		}

		j += kb;
	}
}

/// Solves `A·x = b` in place using factors from [`lu_factor_in_place`]:
/// applies the row swaps to `b`, then unit-lower forward substitution and
/// upper back substitution. `b` is a single column.
pub fn lu_solve_in_place(a: MatRef<'_, f64>, piv: &[usize], b: &mut [f64]) {
	let n = a.nrows();
	assert!(a.ncols() == n && b.len() == n && piv.len() >= n);
	assert!(a.row_stride() == 1);
	let cs = a.col_stride() as usize;
	let base = a.as_ptr();
	unsafe {
		for k in 0..n {
			let p = piv[k];
			if p != k {
				b.swap(k, p);
			}
		}
		// L y = P b (unit lower)
		for k in 0..n {
			let yk = b[k];
			if yk != 0.0 {
				let col = base.add(k * cs);
				axpy(b.as_mut_ptr().add(k + 1), col.add(k + 1), yk, n - k - 1);
			}
		}
		// U x = y
		for k in (0..n).rev() {
			let col = base.add(k * cs);
			let xk = b[k] / *col.add(k);
			b[k] = xk;
			if xk != 0.0 {
				axpy(b.as_mut_ptr(), col, xk, k);
			}
		}
	}
}
