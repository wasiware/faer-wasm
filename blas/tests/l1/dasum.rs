use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn asum_error_bounded() {
	let mut rng = Lcg(7);
	for &n in SIZES {
		let x = rng.vec_f64(n);
		let got = dasum(&x);
		let reference = comp_sum(x.iter().map(|v| v.abs()));
		let tol = f64::EPSILON * (n.max(1) as f64) * reference + f64::MIN_POSITIVE;
		assert!(
			(got - reference).abs() <= tol,
			"dasum n={n}: got {got}, ref {reference}, tol {tol}"
		);
		assert!(got >= 0.0);
	}
}
