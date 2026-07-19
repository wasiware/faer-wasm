use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn dot_error_bounded() {
	let mut rng = Lcg(6);
	for &n in SIZES {
		let x = rng.vec_f32(n);
		let y = rng.vec_f32(n);
		let got = sdot(&x, &y) as f64;
		// NOTE the products are formed in f32 (as the implementation
		// forms them) and only summed in f64 — the reference isolates
		// the summation error, which is what the accumulator layout
		// changes.
		let reference = comp_sum((0..n).map(|i| (x[i] * y[i]) as f64));
		let scale = comp_sum((0..n).map(|i| ((x[i] * y[i]).abs()) as f64));
		let tol = EPS * (n.max(1) as f64) * scale + f32::MIN_POSITIVE as f64;
		assert!(
			(got - reference).abs() <= tol,
			"sdot n={n}: got {got}, ref {reference}, tol {tol}"
		);
	}
}
