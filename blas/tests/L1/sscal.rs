use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn scal_bit_for_bit() {
	let mut rng = Lcg(3);
	for &n in SIZES {
		for alpha in [0.0f32, 1.0, -1.5, 0.33333334, 1e30] {
			let x0 = rng.vec_f32(n);
			let mut x = x0.clone();
			sscal(alpha, &mut x);
			for i in 0..n {
				assert_eq!(x[i].to_bits(), (x0[i] * alpha).to_bits(), "sscal n={n} i={i}");
			}
		}
	}
}
