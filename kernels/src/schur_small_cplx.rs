//! c64 single-shift Hessenberg QR (`zlahqr`-shape) with `want_t`/`Z` — the
//! complex twin of `schur_small.rs`, built for the Schur campaign's
//! decision point (e). Ported from faer's complex `lahqr` at the pinned
//! base (which chases the 2×1 bulge with **Givens rotations**, not
//! LAPACK's 2×1 Householder reflectors — same shifts, same deflation
//! criteria, same exceptional-shift constants, same `rotg` scaling
//! branches, so convergence behavior matches); the inner loops are flat
//! column-major pointer code over `c64`. One deliberate deviation from
//! faer, recorded in the upstream ledger: `rotg(a, 0)` returns `r = a`
//! (LAPACK `zlartg` semantics) where faer returns `r = 1` and would write
//! it over the subdiagonal.

use faer::{c64, MatMut};

use crate::cplx::{cabs, cabs1, cmul_real, conj, csqrt, rotg};

/// eigenvalues of the complex 2×2 [[a00,a01],[a10,a11]] (faer's complex
/// `lahqr_eig22`: abs1-scaled, principal square root)
#[inline]
fn eig22(a00: c64, a01: c64, a10: c64, a11: c64) -> (c64, c64) {
	let s = cabs1(a00) + cabs1(a01) + cabs1(a10) + cabs1(a11);
	if s == 0.0 {
		return (c64::new(0.0, 0.0), c64::new(0.0, 0.0));
	}
	let s_inv = 1.0 / s;
	let a00 = cmul_real(a00, s_inv);
	let a01 = cmul_real(a01, s_inv);
	let a10 = cmul_real(a10, s_inv);
	let a11 = cmul_real(a11, s_inv);
	let tr = cmul_real(a00 + a11, 0.5);
	let det = (a00 - tr) * (a00 - tr) + a01 * a10;
	let rtdisc = csqrt(det);
	(cmul_real(tr + rtdisc, s), cmul_real(tr - rtdisc, s))
}

/// Computes the complex Schur form of an upper-Hessenberg `h` in place.
///
/// - `want_t = true`: on success `h` is the upper triangular `T` (the
///   rotation applies run over the widened ranges). With `want_t = false`
///   only the active block is maintained (`h` is junk on exit, eigenvalues
///   valid) — the instrumentation toggle, as in the real kernel.
/// - `z`: if provided (n×n, pre-initialized — identity or an accumulated
///   `Q` for `A = Z T Zᴴ`), every rotation is applied to its columns.
///
/// Eigenvalues land in `w`. Returns 0 on success, or the failing index+1
/// on non-convergence.
pub fn chqr_schur_in_place(
	h: MatMut<'_, c64>,
	z: Option<MatMut<'_, c64>>,
	w: &mut [c64],
	want_t: bool,
) -> isize {
	let n = h.nrows();
	assert!(h.ncols() == n, "square input required");
	assert!(w.len() >= n);
	assert!(h.row_stride() == 1, "column-major with unit row stride required");
	let cs_h = h.col_stride() as usize;
	let p = h.as_ptr_mut();
	let (zp, cs_z, have_z) = match z {
		Some(z) => {
			assert!(z.nrows() == n && z.ncols() == n, "z must be n×n");
			assert!(z.row_stride() == 1, "z: column-major with unit row stride required");
			let cs = z.col_stride() as usize;
			(z.as_ptr_mut(), cs, true)
		}
		None => (core::ptr::null_mut(), 0usize, false),
	};

	let eps = f64::EPSILON;
	let small_num = f64::MIN_POSITIVE / eps;
	let dat1 = 0.75f64;
	let dat2 = -0.4375f64;
	const NON_CONVERGENCE_LIMIT: usize = 10;

	if n == 0 {
		return 0;
	}
	if n == 1 {
		unsafe {
			w[0] = *p;
		}
		return 0;
	}

	unsafe {
		macro_rules! at {
			($r:expr, $c:expr) => {
				*p.add(($r) + ($c) * cs_h)
			};
		}

		// faer complex lahqr: itmax = max(30, nbits/2)·max(10, nh)·nh with
		// nbits/2 = 26 for c64 — i.e. 30·max(10, n)·n
		let itmax = 30usize.saturating_mul(Ord::max(10, n)).saturating_mul(n);
		let mut k_defl = 0usize;
		let mut istop = n;
		let mut istart = 0usize;

		for iter in 0..itmax {
			if iter + 1 == itmax {
				return istop as isize;
			}
			if istart + 1 >= istop {
				if istart + 1 == istop {
					w[istart] = at!(istart, istart);
				}
				break;
			}

			let (istart_m, istop_m) = if want_t { (0usize, n) } else { (istart, istop) };

			// deflation scan (abs1-based, Ahues–Tisseur secondary test)
			for i in (istart + 1..istop).rev() {
				let sub = at!(i, i - 1);
				if cabs1(sub) < small_num {
					at!(i, i - 1) = c64::new(0.0, 0.0);
					istart = i;
					break;
				}
				let mut tst = cabs1(at!(i - 1, i - 1)) + cabs1(at!(i, i));
				if tst == 0.0 {
					if i >= 2 {
						tst += cabs1(at!(i - 1, i - 2));
					}
					if i + 1 < n {
						tst += cabs1(at!(i + 1, i));
					}
				}
				if cabs1(sub) <= eps * tst {
					let sup = at!(i - 1, i);
					let ab = cabs1(sub).max(cabs1(sup));
					let ba = cabs1(sub).min(cabs1(sup));
					let d = at!(i, i) - at!(i - 1, i - 1);
					let aa = cabs1(at!(i, i)).max(cabs1(d));
					let bb = cabs1(at!(i, i)).min(cabs1(d));
					let s = aa + ab;
					if ba * (ab / s) <= (eps * (bb * (aa / s))).max(small_num) {
						at!(i, i - 1) = c64::new(0.0, 0.0);
						istart = i;
						break;
					}
				}
			}

			// 1×1 deflation (complex: one eigenvalue at a time)
			if istart + 1 >= istop {
				k_defl = 0;
				w[istart] = at!(istart, istart);
				istop = istart;
				istart = 0;
				continue;
			}

			// single shift: eigenvalue of the trailing 2×2 closest to the
			// corner entry, or the exceptional shift every 10 stalled rounds
			let (a00, a01, a10, a11);
			k_defl += 1;
			if k_defl % NON_CONVERGENCE_LIMIT == 0 {
				let mut s = cabs(at!(istop - 1, istop - 2));
				if istop > 2 {
					s += cabs(at!(istop - 2, istop - 3));
				}
				a00 = c64::new(dat1 * s, 0.0) + at!(istop - 1, istop - 1);
				a10 = c64::new(dat2 * s, 0.0);
				a01 = c64::new(s, 0.0);
				a11 = a00;
			} else {
				a00 = at!(istop - 2, istop - 2);
				a01 = at!(istop - 2, istop - 1);
				a10 = at!(istop - 1, istop - 2);
				a11 = at!(istop - 1, istop - 1);
			}
			let (mut s1, s2) = eig22(a00, a01, a10, a11);
			{
				let corner = at!(istop - 1, istop - 1);
				if cabs1(s1 - corner) > cabs1(s2 - corner) {
					s1 = s2;
				}
			}

			// small-subdiagonal early start
			let mut istart2 = istart;
			if istart + 2 < istop {
				for i in (istart + 1..istop - 1).rev() {
					let h00 = at!(i, i) - s1;
					let h10 = at!(i + 1, i);
					let (_, sn, _) = rotg(h00, h10);
					if cabs1(conj(sn) * at!(i, i - 1))
						<= eps * (cabs1(at!(i, i - 1)) + cabs1(at!(i, i + 1)))
					{
						istart2 = i;
						break;
					}
				}
			}

			// the single-shift rotation chase over [istart2, istop)
			for i in istart2..istop - 1 {
				let (c, s);
				if i == istart2 {
					let (rc, rs, _) = rotg(at!(i, i) - s1, at!(i + 1, i));
					c = rc;
					s = rs;
					if i > istart {
						at!(i, i - 1) = cmul_real(at!(i, i - 1), c);
					}
				} else {
					let (rc, rs, r) = rotg(at!(i, i - 1), at!(i + 1, i - 1));
					c = rc;
					s = rs;
					at!(i, i - 1) = r;
					at!(i + 1, i - 1) = c64::new(0.0, 0.0);
				}
				let sc = conj(s);

				// left-apply to rows (i, i+1) over columns i..istop_m:
				// x' = c·x − conj(s)·y ; y' = s·x + c·y
				{
					let mut pj = p.add(i + i * cs_h);
					let mut j = i;
					while j < istop_m {
						let x = *pj;
						let y = *pj.add(1);
						*pj = cmul_real(x, c) - sc * y;
						*pj.add(1) = s * x + cmul_real(y, c);
						pj = pj.add(cs_h);
						j += 1;
					}
				}
				// right-apply (adjoint) to columns (i, i+1) over rows
				// istart_m..min(i+3, istop): x' = c·x − s·y ; y' = conj(s)·x + c·y
				{
					let c0 = p.add(i * cs_h);
					let c1 = p.add((i + 1) * cs_h);
					let jend = Ord::min(i + 3, istop);
					let mut r = istart_m;
					while r < jend {
						let x = *c0.add(r);
						let y = *c1.add(r);
						*c0.add(r) = cmul_real(x, c) - s * y;
						*c1.add(r) = sc * x + cmul_real(y, c);
						r += 1;
					}
				}
				if have_z {
					let z0 = zp.add(i * cs_z);
					let z1 = zp.add((i + 1) * cs_z);
					let mut r = 0usize;
					while r < n {
						let x = *z0.add(r);
						let y = *z1.add(r);
						*z0.add(r) = cmul_real(x, c) - s * y;
						*z1.add(r) = sc * x + cmul_real(y, c);
						r += 1;
					}
				}
			}
		}
	}
	0
}
