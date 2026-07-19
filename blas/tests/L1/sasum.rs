use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn asum_error_bounded() {
	let mut rng = Lcg(7);
	for &n in SIZES {
		let x = rng.vec_f32(n);
		let got = sasum(&x) as f64;
		let reference = comp_sum(x.iter().map(|v| v.abs() as f64));
		let tol = EPS * (n.max(1) as f64) * reference + f32::MIN_POSITIVE as f64;
		assert!(
			(got - reference).abs() <= tol,
			"sasum n={n}: got {got}, ref {reference}, tol {tol}"
		);
		assert!(got >= 0.0);
	}
}
