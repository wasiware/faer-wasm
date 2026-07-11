//! c64 Hessenberg reduction (`zgehd2`-shape) + Q formation (`zunghr`-shape
//! backward accumulation) — the complex twins of `hessenberg.rs`, built for
//! the Schur campaign's decision point (e). Flat scalar complex loops (no
//! explicit simd128 yet — measure first); same structure as the real
//! kernel: per reflector, one gaxpy right-apply pass over all rows and one
//! dot/axpy left-apply pass over the trailing block.
//!
//! Complex Householder conventions (LAPACK `zlarfg`/`zgehd2`): the
//! reflector is `H = I − τ·v·vᴴ` with complex `τ`, `v[0] = 1` implicit,
//! `β` real; the reduction applies `H` from the right with `τ` and from
//! the left with `conj(τ)`, so `A_orig = Q·H·Qᴴ` with
//! `Q = H_0·H_1···H_{k−1}`. Like the real kernels, the `zlarfg` small-β
//! rescaling path is skipped (well-conditioned dense regime).

use faer::{c64, MatMut};

use crate::cplx::{cabs2, conj};

/// Reduces square `A` (n×n c64, column-major, unit row stride) to upper
/// Hessenberg form in place; reflector tails below the first subdiagonal,
/// `tau[j]` the complex reflector scalars (`k = n−2`), `work` ≥ n scratch.
pub fn hessenberg_cplx_factor_in_place(a: MatMut<'_, c64>, tau: &mut [c64], work: &mut [c64]) {
	let n = a.nrows();
	assert!(a.ncols() == n, "square input required");
	let k = n.saturating_sub(2);
	assert!(tau.len() >= k, "tau must hold n-2 scalars");
	assert!(work.len() >= n, "work must hold n scratch scalars");
	assert!(a.row_stride() == 1, "column-major with unit row stride required");
	let cs = a.col_stride() as usize;
	let base = a.as_ptr_mut();
	let w = work.as_mut_ptr();

	unsafe {
		for j in 0..k {
			let col = base.add(j * cs);
			let alpha = *col.add(j + 1);
			let tail = n - j - 2;

			// ‖x‖² over the stored tail
			let mut xnorm_sq = 0.0f64;
			let mut t = 0usize;
			while t < tail {
				xnorm_sq += cabs2(*col.add(j + 2 + t));
				t += 1;
			}
			if xnorm_sq == 0.0 && alpha.im == 0.0 {
				tau[j] = c64::new(0.0, 0.0);
				continue;
			}

			// zlarfg: β = −sign(Re α)·‖(α, x)‖ (real);
			// τ = ((β − Re α)/β, −Im α/β); v = x/(α − β)
			let anorm = libm::sqrt(cabs2(alpha) + xnorm_sq);
			let beta = if alpha.re >= 0.0 { -anorm } else { anorm };
			let tj = c64::new((beta - alpha.re) / beta, -alpha.im / beta);
			let inv = c64::new(1.0, 0.0) / (alpha - c64::new(beta, 0.0));
			let mut t = 0usize;
			while t < tail {
				*col.add(j + 2 + t) *= inv;
				t += 1;
			}
			tau[j] = tj;
			*col.add(j + 1) = c64::new(beta, 0.0);

			// ---- right-apply to all rows: A[:, j+1..n] := A[:, j+1..n]·H.
			// gaxpy pass: w = A·v = A[:, j+1] + Σ_t v[t]·A[:, j+2+t]
			let first = base.add((j + 1) * cs);
			core::ptr::copy_nonoverlapping(first, w, n);
			for t in 0..tail {
				let vt = *col.add(j + 2 + t);
				if vt.re != 0.0 || vt.im != 0.0 {
					let src = base.add((j + 2 + t) * cs);
					let mut r = 0usize;
					while r < n {
						*w.add(r) += vt * *src.add(r);
						r += 1;
					}
				}
			}
			// update pass: A[:, j+1] −= τ·w; A[:, j+2+t] −= (τ·conj(v[t]))·w
			{
				let mut r = 0usize;
				while r < n {
					*first.add(r) -= tj * *w.add(r);
					r += 1;
				}
			}
			for t in 0..tail {
				let vt = *col.add(j + 2 + t);
				if vt.re != 0.0 || vt.im != 0.0 {
					let f = tj * conj(vt);
					let dst = base.add((j + 2 + t) * cs);
					let mut r = 0usize;
					while r < n {
						*dst.add(r) -= f * *w.add(r);
						r += 1;
					}
				}
			}

			// ---- left-apply to the trailing block with conj(τ):
			// per column: s = conj(τ)·(vᴴ·c); c[j+1] −= s; c[j+2..] −= s·v
			let tjc = conj(tj);
			let mut c = j + 1;
			while c < n {
				let ac = base.add(c * cs);
				let mut s = *ac.add(j + 1);
				let mut t = 0usize;
				while t < tail {
					s += conj(*col.add(j + 2 + t)) * *ac.add(j + 2 + t);
					t += 1;
				}
				let s = tjc * s;
				*ac.add(j + 1) -= s;
				let mut t = 0usize;
				while t < tail {
					*ac.add(j + 2 + t) -= s * *col.add(j + 2 + t);
					t += 1;
				}
				c += 1;
			}
		}
	}
}

/// Forms `Q = H_0·H_1···H_{k−1}` from the reflectors stored by
/// [`hessenberg_cplx_factor_in_place`] (`zunghr`-shape backward
/// accumulation, ~4/3·n³ complex flops; columns `0..j+1` stay identity at
/// step `j`, same as the real [`crate::hessenberg::hessenberg_form_q`]).
pub fn hessenberg_cplx_form_q(a: faer::MatRef<'_, c64>, tau: &[c64], q: MatMut<'_, c64>) {
	let n = a.nrows();
	assert!(a.ncols() == n, "square input required");
	assert!(q.nrows() == n && q.ncols() == n, "q must be n×n");
	let k = n.saturating_sub(2);
	assert!(tau.len() >= k, "tau must hold n-2 scalars");
	assert!(a.row_stride() == 1, "column-major with unit row stride required");
	assert!(q.row_stride() == 1, "q: column-major with unit row stride required");
	let acs = a.col_stride() as usize;
	let qcs = q.col_stride() as usize;
	let ap = a.as_ptr();
	let qp = q.as_ptr_mut();

	unsafe {
		for c in 0..n {
			let col = qp.add(c * qcs);
			for r in 0..n {
				*col.add(r) = if r == c { c64::new(1.0, 0.0) } else { c64::new(0.0, 0.0) };
			}
		}
		for j in (0..k).rev() {
			let tj = tau[j];
			if tj.re == 0.0 && tj.im == 0.0 {
				continue;
			}
			let v = ap.add(j * acs);
			let tail = n - j - 2;
			for c in j + 1..n {
				let qc = qp.add(c * qcs);
				// s = τ_j·(vᴴ·Q[j+1.., c])
				let mut s = *qc.add(j + 1);
				let mut t = 0usize;
				while t < tail {
					s += conj(*v.add(j + 2 + t)) * *qc.add(j + 2 + t);
					t += 1;
				}
				let s = tj * s;
				*qc.add(j + 1) -= s;
				let mut t = 0usize;
				while t < tail {
					*qc.add(j + 2 + t) -= s * *v.add(j + 2 + t);
					t += 1;
				}
			}
		}
	}
}
