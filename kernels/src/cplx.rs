//! c64 helpers for the complex Schur kernels (Schur campaign decision
//! point (e), 2026-07-11). Flat scalar complex arithmetic over
//! `faer::c64` (= `num_complex::Complex<f64>`) — deliberately no explicit
//! simd128 yet: the Jacobi-probe discipline says build the bare correct
//! thing, measure on the runner, then decide where simd pays. The O(n³)
//! bulk still routes through faer's c64 gemm (a measured 4.8–5.3× win).

use faer::c64;

/// |re| + |im| — LAPACK `CABS1`, the cheap magnitude used by all the
/// deflation tests
#[inline(always)]
pub fn cabs1(z: c64) -> f64 {
	z.re.abs() + z.im.abs()
}

/// modulus. The kernels only call this on deflation-scan and shift values
/// (never accumulating), and `rotg` does its own scaling, so the naive
/// formula's overflow window (~1e154) is irrelevant in our regime.
#[inline(always)]
pub fn cabs(z: c64) -> f64 {
	libm::sqrt(z.re * z.re + z.im * z.im)
}

#[inline(always)]
pub fn cabs2(z: c64) -> f64 {
	z.re * z.re + z.im * z.im
}

#[inline(always)]
pub fn conj(z: c64) -> c64 {
	c64::new(z.re, -z.im)
}

#[inline(always)]
pub fn cmul_real(z: c64, s: f64) -> c64 {
	c64::new(z.re * s, z.im * s)
}

/// principal complex square root
#[inline]
pub fn csqrt(z: c64) -> c64 {
	if z.re == 0.0 && z.im == 0.0 {
		return c64::new(0.0, 0.0);
	}
	let r = cabs(z);
	let re = libm::sqrt(0.5 * (r + z.re));
	let im = libm::sqrt(0.5 * (r - z.re));
	c64::new(re, if z.im < 0.0 { -im } else { im })
}

/// Complex Givens rotation, `zlartg`-shape — a port of faer's
/// `JacobiRotation::rotg` complex branch at the pinned base (same scaled
/// branches, so convergence behavior matches), returning `(c, s, r)` with
/// `c` real. Applying the rotation to `(a, b)` as
/// `x' = c·x − conj(s)·y, y' = s·x + c·y` gives `(r, 0)`.
///
/// One deliberate deviation, recorded in the ROADMAP upstream ledger:
/// faer returns `r = 1` when `b == 0` (its `lahqr` then writes that `r`
/// over the subdiagonal — wrong for a measure-zero input class); LAPACK
/// `zlartg` returns `r = a`, and so do we.
pub fn rotg(a: c64, b: c64) -> (f64, c64, c64) {
	if b.re == 0.0 && b.im == 0.0 {
		return (1.0, c64::new(0.0, 0.0), a);
	}
	let eps = f64::EPSILON;
	let sml = f64::MIN_POSITIVE;
	let big = 1.0 / sml;
	let rtmin = libm::sqrt(sml / eps);
	let rtmax = 1.0 / rtmin;
	let (c, s, r);
	if a.re == 0.0 && a.im == 0.0 {
		c = 0.0;
		let g1 = b.re.abs().max(b.im.abs());
		if g1 > rtmin && g1 < rtmax {
			let g2 = cabs2(b);
			let d = libm::sqrt(g2);
			s = cmul_real(conj(b), 1.0 / d);
			r = c64::new(d, 0.0);
		} else {
			let u = g1.max(sml).min(big);
			let uu = 1.0 / u;
			let gs = cmul_real(b, uu);
			let g2 = cabs2(gs);
			let d = libm::sqrt(g2);
			s = cmul_real(conj(gs), 1.0 / d);
			r = c64::new(d * u, 0.0);
		}
	} else {
		let f1 = a.re.abs().max(a.im.abs());
		let g1 = b.re.abs().max(b.im.abs());
		if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
			let f2 = cabs2(a);
			let g2 = cabs2(b);
			let h2 = f2 + g2;
			let d = if f2 > rtmin && h2 < rtmax {
				libm::sqrt(f2 * h2)
			} else {
				libm::sqrt(f2) * libm::sqrt(h2)
			};
			let p = 1.0 / d;
			c = f2 * p;
			s = conj(b) * cmul_real(a, p);
			r = cmul_real(a, h2 * p);
		} else {
			let u = f1.max(g1).max(sml).min(big);
			let uu = 1.0 / u;
			let gs = cmul_real(b, uu);
			let g2 = cabs2(gs);
			let (f2, h2, w);
			let fs;
			if f1 * uu < rtmin {
				let v = f1.max(sml).min(big);
				let vv = 1.0 / v;
				w = v * uu;
				fs = cmul_real(a, vv);
				f2 = cabs2(fs);
				h2 = (f2 * w) * w + g2;
			} else {
				w = 1.0;
				fs = cmul_real(a, uu);
				f2 = cabs2(fs);
				h2 = f2 + g2;
			}
			let d = if f2 > rtmin && h2 < rtmax {
				libm::sqrt(f2 * h2)
			} else {
				libm::sqrt(f2) * libm::sqrt(h2)
			};
			let p = 1.0 / d;
			c = (f2 * p) * w;
			s = conj(gs) * cmul_real(fs, p);
			r = cmul_real(cmul_real(fs, h2 * p), u);
		}
	}
	// faer stores the rotation as {c, -conj(s)} and its left-apply computes
	// x' = c·x − conj(s_field)·y = c·x + s·y … our apply convention above
	// absorbs that: return s_field = −conj(s)
	(c, -conj(s), r)
}
