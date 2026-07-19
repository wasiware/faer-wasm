use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn swap_bit_for_bit() {
	let mut rng = Lcg(2);
	for &n in SIZES {
		let x0 = rng.vec_f32(n);
		let y0 = rng.vec_f32(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		sswap(&mut x, &mut y);
		for i in 0..n {
			assert_eq!(x[i].to_bits(), y0[i].to_bits(), "sswap n={n} i={i}");
			assert_eq!(y[i].to_bits(), x0[i].to_bits(), "sswap n={n} i={i}");
		}
	}
}
