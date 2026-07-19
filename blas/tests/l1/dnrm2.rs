use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn nrm2_error_bounded() {
	let mut rng = Lcg(8);
	for &n in SIZES {
		let x = rng.vec_f64(n);
		let got = dnrm2(&x);
		let m = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
		let reference = if m == 0.0 {
			0.0
		} else {
			m * comp_sum(x.iter().map(|v| (v / m) * (v / m))).sqrt()
		};
		let tol = f64::EPSILON * (n.max(1) as f64) * reference.max(f64::MIN_POSITIVE);
		assert!(
			(got - reference).abs() <= tol,
			"dnrm2 n={n}: got {got}, ref {reference}, tol {tol}"
		);
	}
}

#[test]
fn nrm2_overflow_underflow_guards() {
	// naive sum of squares overflows: values ~1e300
	let big = vec![1e300, -1e300, 1e300];
	let got = dnrm2(&big);
	let want = 1e300 * 3.0f64.sqrt();
	assert!((got - want).abs() <= 1e287, "overflow rescue: got {got}, want {want}");
	assert!(got.is_finite());

	// naive squares underflow to zero: values ~1e-300
	let tiny = vec![3e-300, 4e-300];
	let got = dnrm2(&tiny);
	let want = 5e-300;
	assert!(
		(got - want).abs() <= 1e-313,
		"underflow rescue: got {got}, want {want}"
	);
	assert!(got > 0.0);

	// mixed sizes across the rescue boundary, checked against the scaled
	// reference
	let mixed = vec![1e300, 1.0, 1e-300, -2e299];
	let m = 1e300f64;
	let want = m * comp_sum(mixed.iter().map(|v| (v / m) * (v / m))).sqrt();
	let got = dnrm2(&mixed);
	assert!((got - want).abs() <= 1e287, "mixed rescue: got {got}, want {want}");

	// exact zeros and empty
	assert_eq!(dnrm2(&[]), 0.0);
	assert_eq!(dnrm2(&[0.0, -0.0, 0.0]), 0.0);
	// infinity in, infinity out
	assert_eq!(dnrm2(&[1.0, f64::INFINITY]), f64::INFINITY);
}
