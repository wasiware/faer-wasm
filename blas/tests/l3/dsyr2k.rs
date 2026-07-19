use super::common::*;
use faer_wasm_blas::l3::*;

#[test]
fn syr2k_bit_for_bit() {
	let mut rng = Lcg(33);
	for &n in NS {
		let k = (n / 2).max(1);
		for upper in [true, false] {
			let (acs, bcs, ccs) = (n + 1, n + 2, n + 1);
			let a = rng.mat_f64(n, k, acs);
			let b = rng.mat_f64(n, k, bcs);
			let c0 = rng.mat_f64(n, n, ccs);


			let mut c = c0.clone();
			dsyr2k(0.5, n, k, &a, acs, &b, bcs, 0.25, &mut c, ccs, upper);
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					cr[j * ccs + i] *= 0.25;
				}
				for l in 0..k {
					let tb = 0.5 * b[l * bcs + j];
					for i in lo..hi {
						cr[j * ccs + i] += a[l * acs + i] * tb;
					}
					let ta = 0.5 * a[l * acs + j];
					for i in lo..hi {
						cr[j * ccs + i] += b[l * bcs + i] * ta;
					}
				}
			}
			for j in 0..n {
				for i in 0..n {
					assert_eq!(c[j * ccs + i].to_bits(), cr[j * ccs + i].to_bits(), "dsyr2k ({i},{j})");
				}
			}
		}
	}
}
