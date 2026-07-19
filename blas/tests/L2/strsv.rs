use super::common::*;
use faer_wasm_blas::L1::sasum;
use faer_wasm_blas::L2::*;

#[test]
fn trsv_bit_for_bit_and_residual() {
	let mut rng = Lcg(27);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				// diagonally dominant triangle: solves stay well-conditioned
				let mut a = rng.mat_f32(n, n, cs);
				for j in 0..n {
					a[j * cs + j] = 2.0 * (n as f32) + 1.0 + j as f32;
				}
				let b = rng.vec_f32(n);
				let mut x = b.clone();
				strsv(n, &a, cs, upper, unit, &mut x);

				// same-order scalar replay: bit-for-bit
				let mut xr = b.clone();
				if upper {
					for j in (0..n).rev() {
						if !unit {
							xr[j] /= a[j * cs + j];
						}
						let t = xr[j];
						for i in 0..j {
							xr[i] += a[j * cs + i] * -t;
						}
					}
				} else {
					for j in 0..n {
						if !unit {
							xr[j] /= a[j * cs + j];
						}
						let t = xr[j];
						for i in j + 1..n {
							xr[i] += a[j * cs + i] * -t;
						}
					}
				}
				for i in 0..n {
					assert_eq!(
						x[i].to_bits(),
						xr[i].to_bits(),
						"strsv bits upper={upper} unit={unit} n={n} i={i}"
					);
				}

				// independent residual: A·x must reproduce b
				for i in 0..n {
					let ax = comp_sum32((0..n).map(|j| {
						let in_tri = if upper { i <= j } else { i >= j };
						if !in_tri {
							return 0.0;
						}
						let aij = if unit && i == j { 1.0 } else { a[j * cs + i] };
						aij * x[j]
					}));
					let scale = sasum(&x) as f64 * (2.0 * n as f64 + n as f64) + b[i].abs() as f64;
					let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 8.0 * scale + 1e-40;
					assert!(
						(ax - b[i] as f64).abs() <= tol,
						"strsv residual upper={upper} unit={unit} n={n} i={i}: {ax} vs {}",
						b[i]
					);
				}
			}
		}
	}
}
