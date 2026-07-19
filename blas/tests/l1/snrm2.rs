use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn nrm2_error_bounded() {
	let mut rng = Lcg(8);
	for &n in SIZES {
		let x = rng.vec_f32(n);
		let got = snrm2(&x) as f64;
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
			"snrm2 n={n}: got {got}, ref {reference}, tol {tol}"
		);
	}
}

#[test]
fn nrm2_overflow_underflow_guards() {
	// naive sum of squares overflows f32: values ~1e30 (squares 1e60)
	let big = vec![1e30f32, -1e30, 1e30];
	let got = snrm2(&big);
	let want = 1e30f32 * 3.0f32.sqrt();
	assert!((got - want).abs() <= 1e24, "overflow rescue: got {got}, want {want}");
	assert!(got.is_finite());

	// naive squares underflow to zero: values ~1e-30
	let tiny = vec![3e-30f32, 4e-30];
	let got = snrm2(&tiny);
	let want = 5e-30f32;
	assert!((got - want).abs() <= 1e-35, "underflow rescue: got {got}, want {want}");
	assert!(got > 0.0);

	// mixed sizes across the rescue boundary, checked against the
	// scaled f64 reference
	let mixed = vec![1e30f32, 1.0, 1e-30, -2e29];
	let m = 1e30f64;
	let want =
		m * comp_sum(mixed.iter().map(|v| (*v as f64 / m) * (*v as f64 / m))).sqrt();
	let got = snrm2(&mixed) as f64;
	assert!((got - want).abs() <= 1e24, "mixed rescue: got {got}, want {want}");

	// exact zeros and empty
	assert_eq!(snrm2(&[]), 0.0);
	assert_eq!(snrm2(&[0.0, -0.0, 0.0]), 0.0);
	// infinity in, infinity out
	assert_eq!(snrm2(&[1.0, f32::INFINITY]), f32::INFINITY);
}
