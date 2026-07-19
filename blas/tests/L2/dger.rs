use super::common::*;
use faer_wasm_blas::L2::*;

#[test]
fn ger_bit_for_bit() {
	let mut rng = Lcg(23);
	for &(m, n) in SHAPES {
		let cs = m + 1;
		let a0 = rng.mat_f64(m, n, cs);
		let x = rng.vec_f64(m);
		let y = rng.vec_f64(n);
		let mut a = a0.clone();
		dger(-1.3, m, n, &mut a, cs, &x, &y);
		for j in 0..n {
			let t = -1.3 * y[j];
			for i in 0..m {
				let want = a0[j * cs + i] + x[i] * t;
				assert_eq!(a[j * cs + i].to_bits(), want.to_bits(), "dger {m}x{n} ({i},{j})");
			}
		}
	}
}
