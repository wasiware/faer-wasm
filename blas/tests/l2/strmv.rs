use super::common::*;
use faer_wasm_blas::l2::*;

#[test]
fn trmv_bit_for_bit_all_variants() {
	let mut rng = Lcg(26);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				let a = rng.mat_f32(n, n, cs);
				let x0 = rng.vec_f32(n);
				let mut x = x0.clone();
				strmv(n, &a, cs, upper, unit, &mut x);
				// same-order scalar replay
				let mut xr = x0.clone();
				if upper {
					for j in 0..n {
						let t = xr[j];
						for i in 0..j {
							xr[i] += a[j * cs + i] * t;
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				} else {
					for j in (0..n).rev() {
						let t = xr[j];
						for i in j + 1..n {
							xr[i] += a[j * cs + i] * t;
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				}
				for i in 0..n {
					assert_eq!(
						x[i].to_bits(),
						xr[i].to_bits(),
						"strmv upper={upper} unit={unit} n={n} i={i}"
					);
				}
			}
		}
	}
}
