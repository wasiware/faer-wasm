use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn dot_error_bounded() {
	let mut rng = Lcg(6);
	for &n in SIZES {
		let x = rng.vec_f64(n);
		let y = rng.vec_f64(n);
		let got = ddot(&x, &y);
		let reference = comp_sum((0..n).map(|i| x[i] * y[i]));
		let scale = comp_sum((0..n).map(|i| (x[i] * y[i]).abs()));
		let tol = f64::EPSILON * (n.max(1) as f64) * scale + f64::MIN_POSITIVE;
		assert!(
			(got - reference).abs() <= tol,
			"ddot n={n}: got {got}, ref {reference}, tol {tol}"
		);
	}
}
