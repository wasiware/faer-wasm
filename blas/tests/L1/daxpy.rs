use super::common::*;
use faer_wasm_blas::L1::*;

#[test]
fn axpy_bit_for_bit() {
	let mut rng = Lcg(4);
	for &n in SIZES {
		for alpha in [0.0, 1.0, -2.5, 0.1] {
			let x = rng.vec_f64(n);
			let y0 = rng.vec_f64(n);
			let mut y = y0.clone();
			daxpy(alpha, &x, &mut y);
			for i in 0..n {
				let want = y0[i] + x[i] * alpha;
				assert_eq!(y[i].to_bits(), want.to_bits(), "daxpy n={n} i={i}");
			}
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn axpy_length_mismatch_panics() {
	daxpy(1.0, &[1.0, 2.0], &mut [1.0]);
}
