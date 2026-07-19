use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn copy_bit_for_bit() {
	let mut rng = Lcg(1);
	for &n in SIZES {
		let x = rng.vec_f64(n);
		let mut y = vec![0.0; n];
		dcopy(&x, &mut y);
		for i in 0..n {
			assert_eq!(x[i].to_bits(), y[i].to_bits(), "dcopy n={n} i={i}");
		}
	}
}
