use super::common::*;
use faer_wasm_blas::L2::*;

#[test]
fn syr_bit_for_bit() {
	let mut rng = Lcg(25);
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 2;
			let a0 = rng.mat_f64(n, n, cs);
			let x = rng.vec_f64(n);

			let mut a = a0.clone();
			dsyr(0.6, n, &mut a, cs, upper, &x);
			for j in 0..n {
				let t = 0.6 * x[j];
				for i in 0..n {
					let stored = if upper { i <= j } else { i >= j };
					let want = if stored { a0[j * cs + i] + x[i] * t } else { a0[j * cs + i] };
					assert_eq!(a[j * cs + i].to_bits(), want.to_bits(), "dsyr n={n} ({i},{j})");
				}
			}
		}
	}
}
