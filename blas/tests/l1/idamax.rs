use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn iamax_exact_semantics() {
	// first-occurrence tie-breaking, negatives, single element, empty
	assert_eq!(idamax(&[]), 0);
	assert_eq!(idamax(&[7.0]), 0);
	assert_eq!(idamax(&[1.0, -3.0, 3.0, 2.0]), 1, "tie: first occurrence");
	assert_eq!(idamax(&[2.0, 2.0, 2.0]), 0, "all equal");
	assert_eq!(idamax(&[0.0, 0.0]), 0, "all zero");
	assert_eq!(idamax(&[-0.5, 0.25, -0.75]), 2);

	// agreement with the plain-loop definition on random data, all sizes
	let mut rng = Lcg(9);
	for &n in SIZES {
		let x = rng.vec_f64(n);
		let got = idamax(&x);
		let mut m = -1.0f64;
		let mut mi = 0usize;
		for (i, v) in x.iter().enumerate() {
			if v.abs() > m {
				m = v.abs();
				mi = i;
			}
		}
		assert_eq!(got, mi, "idamax n={n}");
	}
}
