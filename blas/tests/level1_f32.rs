//! f32 Level 1 correctness — the f64 suite cloned per the testing
//! contract (../README.md): elementwise streams bit-for-bit against
//! the f32 scalar definition; reductions error-bounded against an
//! f64 compensated-summation reference (the higher-precision
//! reference the contract asks for); iamax's index exact; nrm2's
//! over/underflow guards exercised at f32 ranges (~1e±30); rotg
//! against its defining identities at f32 tolerances.

use faer_wasm_blas::f32::level1::*;

struct Lcg(u64);
impl Lcg {
	fn next(&mut self) -> f32 {
		self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		let bits = (self.0 >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
		(4.0 * bits - 2.0) as f32
	}
	fn vec(&mut self, n: usize) -> Vec<f32> {
		(0..n).map(|_| self.next()).collect()
	}
}

// Sizes that exercise the empty case, the pure-tail path (< 8/16),
// unroll boundaries, and odd tails at the f32 strides.
const SIZES: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 33, 100, 257, 1000];

// Neumaier compensated summation in f64 — the higher-precision
// reference for the f32 reduction bounds.
fn comp_sum(it: impl Iterator<Item = f64>) -> f64 {
	let mut s = 0.0f64;
	let mut c = 0.0f64;
	for v in it {
		let t = s + v;
		if s.abs() >= v.abs() {
			c += (s - t) + v;
		} else {
			c += (v - t) + s;
		}
		s = t;
	}
	s + c
}

const EPS: f64 = f32::EPSILON as f64;

// ---- elementwise streams: bit-for-bit ----

#[test]
fn copy_bit_for_bit() {
	let mut rng = Lcg(1);
	for &n in SIZES {
		let x = rng.vec(n);
		let mut y = vec![0.0f32; n];
		copy(&x, &mut y);
		for i in 0..n {
			assert_eq!(x[i].to_bits(), y[i].to_bits(), "copy n={n} i={i}");
		}
	}
}

#[test]
fn swap_bit_for_bit() {
	let mut rng = Lcg(2);
	for &n in SIZES {
		let x0 = rng.vec(n);
		let y0 = rng.vec(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		swap(&mut x, &mut y);
		for i in 0..n {
			assert_eq!(x[i].to_bits(), y0[i].to_bits(), "swap n={n} i={i}");
			assert_eq!(y[i].to_bits(), x0[i].to_bits(), "swap n={n} i={i}");
		}
	}
}

#[test]
fn scal_bit_for_bit() {
	let mut rng = Lcg(3);
	for &n in SIZES {
		for alpha in [0.0f32, 1.0, -1.5, 0.33333334, 1e30] {
			let x0 = rng.vec(n);
			let mut x = x0.clone();
			scal(alpha, &mut x);
			for i in 0..n {
				assert_eq!(x[i].to_bits(), (x0[i] * alpha).to_bits(), "scal n={n} i={i}");
			}
		}
	}
}

#[test]
fn axpy_bit_for_bit() {
	let mut rng = Lcg(4);
	for &n in SIZES {
		for alpha in [0.0f32, 1.0, -2.5, 0.1] {
			let x = rng.vec(n);
			let y0 = rng.vec(n);
			let mut y = y0.clone();
			axpy(alpha, &x, &mut y);
			for i in 0..n {
				let want = y0[i] + x[i] * alpha;
				assert_eq!(y[i].to_bits(), want.to_bits(), "axpy n={n} i={i}");
			}
		}
	}
}

#[test]
fn rot_bit_for_bit() {
	let mut rng = Lcg(5);
	let (c, s) = (0.8f32, 0.6f32);
	for &n in SIZES {
		let x0 = rng.vec(n);
		let y0 = rng.vec(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		rot(&mut x, &mut y, c, s);
		for i in 0..n {
			let wx = x0[i] * c + y0[i] * s;
			let wy = y0[i] * c - x0[i] * s;
			assert_eq!(x[i].to_bits(), wx.to_bits(), "rot x n={n} i={i}");
			assert_eq!(y[i].to_bits(), wy.to_bits(), "rot y n={n} i={i}");
		}
	}
}

// ---- reduction streams: error-bounded vs f64 compensated reference ----

#[test]
fn dot_error_bounded() {
	let mut rng = Lcg(6);
	for &n in SIZES {
		let x = rng.vec(n);
		let y = rng.vec(n);
		let got = dot(&x, &y) as f64;
		// NOTE the products are formed in f32 (as the implementation
		// forms them) and only summed in f64 — the reference isolates
		// the summation error, which is what the accumulator layout
		// changes.
		let reference = comp_sum((0..n).map(|i| (x[i] * y[i]) as f64));
		let scale = comp_sum((0..n).map(|i| ((x[i] * y[i]).abs()) as f64));
		let tol = EPS * (n.max(1) as f64) * scale + f32::MIN_POSITIVE as f64;
		assert!(
			(got - reference).abs() <= tol,
			"dot n={n}: got {got}, ref {reference}, tol {tol}"
		);
	}
}

#[test]
fn asum_error_bounded() {
	let mut rng = Lcg(7);
	for &n in SIZES {
		let x = rng.vec(n);
		let got = asum(&x) as f64;
		let reference = comp_sum(x.iter().map(|v| v.abs() as f64));
		let tol = EPS * (n.max(1) as f64) * reference + f32::MIN_POSITIVE as f64;
		assert!(
			(got - reference).abs() <= tol,
			"asum n={n}: got {got}, ref {reference}, tol {tol}"
		);
		assert!(got >= 0.0);
	}
}

#[test]
fn nrm2_error_bounded() {
	let mut rng = Lcg(8);
	for &n in SIZES {
		let x = rng.vec(n);
		let got = nrm2(&x) as f64;
		let m = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
		let reference = if m == 0.0 {
			0.0
		} else {
			let md = m as f64;
			md * comp_sum(x.iter().map(|v| {
				let s = (*v / m) as f64;
				s * s
			}))
			.sqrt()
		};
		let tol = EPS * (n.max(1) as f64) * reference.max(f32::MIN_POSITIVE as f64);
		assert!(
			(got - reference).abs() <= tol,
			"nrm2 n={n}: got {got}, ref {reference}, tol {tol}"
		);
	}
}

#[test]
fn nrm2_overflow_underflow_guards() {
	// naive sum of squares overflows f32: values ~1e30 (squares 1e60)
	let big = vec![1e30f32, -1e30, 1e30];
	let got = nrm2(&big);
	let want = 1e30f32 * 3.0f32.sqrt();
	assert!((got - want).abs() <= 1e24, "overflow rescue: got {got}, want {want}");
	assert!(got.is_finite());

	// naive squares underflow to zero: values ~1e-30
	let tiny = vec![3e-30f32, 4e-30];
	let got = nrm2(&tiny);
	let want = 5e-30f32;
	assert!((got - want).abs() <= 1e-35, "underflow rescue: got {got}, want {want}");
	assert!(got > 0.0);

	// mixed sizes across the rescue boundary, checked against the
	// scaled f64 reference
	let mixed = vec![1e30f32, 1.0, 1e-30, -2e29];
	let m = 1e30f64;
	let want =
		m * comp_sum(mixed.iter().map(|v| (*v as f64 / m) * (*v as f64 / m))).sqrt();
	let got = nrm2(&mixed) as f64;
	assert!((got - want).abs() <= 1e24, "mixed rescue: got {got}, want {want}");

	// exact zeros and empty
	assert_eq!(nrm2(&[]), 0.0);
	assert_eq!(nrm2(&[0.0, -0.0, 0.0]), 0.0);
	// infinity in, infinity out
	assert_eq!(nrm2(&[1.0, f32::INFINITY]), f32::INFINITY);
}

// ---- iamax: exact index semantics ----

#[test]
fn iamax_exact_semantics() {
	assert_eq!(iamax(&[]), 0);
	assert_eq!(iamax(&[7.0]), 0);
	assert_eq!(iamax(&[1.0, -3.0, 3.0, 2.0]), 1, "tie: first occurrence");
	assert_eq!(iamax(&[2.0, 2.0, 2.0]), 0, "all equal");
	assert_eq!(iamax(&[0.0, 0.0]), 0, "all zero");
	assert_eq!(iamax(&[-0.5, 0.25, -0.75]), 2);

	let mut rng = Lcg(9);
	for &n in SIZES {
		let x = rng.vec(n);
		let got = iamax(&x);
		let mut m = -1.0f32;
		let mut mi = 0usize;
		for (i, v) in x.iter().enumerate() {
			if v.abs() > m {
				m = v.abs();
				mi = i;
			}
		}
		assert_eq!(got, mi, "iamax n={n}");
	}
}

// ---- rotg: defining identities + reference edge cases ----

#[test]
fn rotg_identities() {
	let cases = [
		(3.0f32, 4.0f32),
		(4.0, 3.0),
		(-3.0, 4.0),
		(3.0, -4.0),
		(-3.0, -4.0),
		(5.0, 0.0),
		(0.0, 5.0),
		(0.0, -5.0),
		(1e30, 1e30),   // would overflow unguarded (squares > f32::MAX)
		(1e-30, 1e-30), // r² would underflow unguarded (normal-range inputs)
		(1.0, 1e-20),
	];
	for (a, b) in cases {
		let g = rotg(a, b);
		let hyp = (a / g.r).hypot(b / g.r);
		assert!((hyp - 1.0).abs() < 5e-6, "({a},{b}): c²+s² = {hyp}");
		let r1 = g.c * a + g.s * b;
		let z = g.c * b - g.s * a;
		assert!(
			(r1 - g.r).abs() <= 5e-6 * g.r.abs().max(f32::MIN_POSITIVE),
			"({a},{b}): c·a+s·b = {r1}, r = {}",
			g.r
		);
		assert!(
			z.abs() <= 5e-6 * g.r.abs().max(f32::MIN_POSITIVE),
			"({a},{b}): residual {z}"
		);
		let roe = if a.abs() > b.abs() { a } else { b };
		assert_eq!(g.r < 0.0, roe < 0.0, "({a},{b}): sign of r");
	}
	// subnormal inputs (f32 subnormals below ~1.2e-38): reference srotg
	// legitimately loses precision — require only a sane, finite result
	let g = rotg(1e-42, 1e-42);
	assert!(g.r.is_finite() && g.r > 0.0);
	assert!((g.c * g.c + g.s * g.s - 1.0).abs() < 1e-2, "subnormal: c²+s² far off");

	let g = rotg(0.0, 0.0);
	assert_eq!((g.c, g.s, g.r), (1.0, 0.0, 0.0));
	let g = rotg(3.0, 4.0);
	assert!((g.r - 5.0).abs() < 1e-6 && (g.c - 0.6).abs() < 1e-6 && (g.s - 0.8).abs() < 1e-6);
}

// ---- panics on length mismatch (the safe-API contract) ----

#[test]
#[should_panic(expected = "length mismatch")]
fn axpy_length_mismatch_panics() {
	axpy(1.0, &[1.0, 2.0], &mut [1.0]);
}
