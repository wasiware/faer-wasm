use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn scal_bit_for_bit() {
	let mut rng = Lcg(3);
	for &n in SIZES {
		for alpha in [0.0, 1.0, -1.5, 0.3333333333333333, 1e100] {
			let x0 = rng.vec_f64(n);
			let mut x = x0.clone();
			dscal(alpha, &mut x);
			for i in 0..n {
				assert_eq!(x[i].to_bits(), (x0[i] * alpha).to_bits(), "dscal n={n} i={i}");
			}
		}
	}
}
