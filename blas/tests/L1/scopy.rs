use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn copy_bit_for_bit() {
	let mut rng = Lcg(1);
	for &n in SIZES {
		let x = rng.vec_f32(n);
		let mut y = vec![0.0f32; n];
		scopy(&x, &mut y);
		for i in 0..n {
			assert_eq!(x[i].to_bits(), y[i].to_bits(), "scopy n={n} i={i}");
		}
	}
}
