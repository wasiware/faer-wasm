use super::common::*;
use faer_wasm_blas::l1::*;

#[test]
fn rot_bit_for_bit() {
	let mut rng = Lcg(5);
	let (c, s) = (0.8, 0.6);
	for &n in SIZES {
		let x0 = rng.vec_f64(n);
		let y0 = rng.vec_f64(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		drot(&mut x, &mut y, c, s);
		for i in 0..n {
			let wx = x0[i] * c + y0[i] * s;
			let wy = y0[i] * c - x0[i] * s;
			assert_eq!(x[i].to_bits(), wx.to_bits(), "drot x n={n} i={i}");
			assert_eq!(y[i].to_bits(), wy.to_bits(), "drot y n={n} i={i}");
		}
	}
}
