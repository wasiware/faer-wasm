use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn iamax_exact_semantics() {
	assert_eq!(isamax(&[]), 0);
	assert_eq!(isamax(&[7.0]), 0);
	assert_eq!(isamax(&[1.0, -3.0, 3.0, 2.0]), 1, "tie: first occurrence");
	assert_eq!(isamax(&[2.0, 2.0, 2.0]), 0, "all equal");
	assert_eq!(isamax(&[0.0, 0.0]), 0, "all zero");
	assert_eq!(isamax(&[-0.5, 0.25, -0.75]), 2);

	let mut rng = Lcg(9);
	for &n in SIZES {
		let x = rng.vec_f32(n);
		let got = isamax(&x);
		let mut m = -1.0f32;
		let mut mi = 0usize;
		for (i, v) in x.iter().enumerate() {
			if v.abs() > m {
				m = v.abs();
				mi = i;
			}
		}
		assert_eq!(got, mi, "isamax n={n}");
	}
}
