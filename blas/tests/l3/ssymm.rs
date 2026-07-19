use super::common::*;
use faer_wasm_blas::l3::*;

#[test]
fn symm_both_sides_bounded() {
	let mut rng = Lcg(32);
	for &n in NS {
		let m = n; // square B keeps the reference simple
		for upper in [true, false] {
			let acs = n + 1;
			// full symmetric ground truth, one triangle exposed
			let mut full = vec![0.0; n * n];
			for j in 0..n {
				for i in 0..=j {
					let v = rng.next_f32();
					full[j * n + i] = v;
					full[i * n + j] = v;
				}
			}
			let mut a = vec![f32::NAN; if n == 0 { 0 } else { acs * (n - 1) + n }];
			for j in 0..n {
				for i in 0..n {
					if if upper { i <= j } else { i >= j } {
						a[j * acs + i] = full[j * n + i];
					}
				}
			}
			let (bcs, ccs) = (m + 2, m + 1);
			let b = rng.mat_f32(m, n, bcs);
			let c0 = rng.mat_f32(m, n, ccs);

			let mut c = c0.clone();
			ssymm_left(0.8, m, n, &a, acs, upper, &b, bcs, 0.3, &mut c, ccs);
			for j in 0..n {
				for i in 0..m {
					let want = 0.8 * comp_sum32((0..n).map(|l| full[l * n + i] * b[j * bcs + l]))
						+ (0.3 * c0[j * ccs + i]) as f64;
					let scale = comp_sum32((0..n).map(|l| (full[l * n + i] * b[j * bcs + l]).abs()))
						+ c0[j * ccs + i].abs() as f64;
					let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 8.0 * scale + 1e-40;
					assert!((c[j * ccs + i] as f64 - want).abs() <= tol, "ssymm_left ({i},{j})");
				}
			}

			let mut c = c0.clone();
			ssymm_right(0.8, m, n, &a, acs, upper, &b, bcs, 0.3, &mut c, ccs);
			for j in 0..n {
				for i in 0..m {
					let want = 0.8 * comp_sum32((0..n).map(|l| b[l * bcs + i] * full[j * n + l]))
						+ (0.3 * c0[j * ccs + i]) as f64;
					let scale = comp_sum32((0..n).map(|l| (b[l * bcs + i] * full[j * n + l]).abs()))
						+ c0[j * ccs + i].abs() as f64;
					let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 8.0 * scale + 1e-40;
					assert!((c[j * ccs + i] as f64 - want).abs() <= tol, "ssymm_right ({i},{j})");
				}
			}
			// same-order scalar replay (β scale, then k ascending: one
			// α·a rounding + one mul-add per element — the fan-out
			// grouping does not change any element's sequence)
			let mut cr = c0.clone();
			for j in 0..n {
				for i in 0..m {
					cr[j * ccs + i] *= 0.3;
				}
				for k in 0..n {
					let t = 0.8 * full[j * n + k];
					for i in 0..m {
						cr[j * ccs + i] += b[k * bcs + i] * t;
					}
				}
			}
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c[j * ccs + i].to_bits(),
						cr[j * ccs + i].to_bits(),
						"ssymm_right bits n={n} ({i},{j})"
					);
				}
			}
		}
	}
}
