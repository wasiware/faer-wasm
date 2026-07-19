use super::common::*;
use faer_wasm_blas::l3::*;

#[test]
fn trsm_both_sides_residual() {
	let mut rng = Lcg(35);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				let mut a = rng.mat_f32(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = 2.0 * (n as f32) + 1.0 + j as f32;
				}
				let bcs = m + 1;
				let b0 = rng.mat_f32(m, n, bcs);
				let tri_at = |i: usize, j: usize| -> f32 {
					let in_tri = if upper { i <= j } else { i >= j };
					if !in_tri {
						0.0
					} else if unit && i == j {
						1.0
					} else {
						a[j * acs + i]
					}
				};
				let bound = 3.0 * (n as f32 + 1.0);

				// left: A·X = α·B0
				let mut x = b0.clone();
				strsm_left(0.9, m, n, &a, acs, upper, unit, &mut x, bcs);
				for j in 0..n {
					for i in 0..m {
						let ax = comp_sum32((0..n).map(|l| tri_at(i, l) * x[j * bcs + l]));
						let want = (0.9 * b0[j * bcs + i]) as f64;
						let xmax = (0..n).fold(0.0f32, |acc, l| acc.max(x[j * bcs + l].abs()));
						let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 8.0 * bound as f64 * xmax as f64 + 1e-40;
						assert!(
							(ax - want).abs() <= tol,
							"strsm_left residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				// right: X·A = α·B0
				let mut x = b0.clone();
				strsm_right(0.9, m, n, &a, acs, upper, unit, &mut x, bcs);
				for j in 0..n {
					for i in 0..m {
						let xa = comp_sum32((0..n).map(|l| x[l * bcs + i] * tri_at(l, j)));
						let want = (0.9 * b0[j * bcs + i]) as f64;
						let xmax = (0..n).fold(0.0f32, |acc, l| acc.max(x[l * bcs + i].abs()));
						let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 8.0 * bound as f64 * xmax as f64 + 1e-40;
						assert!(
							(xa - want).abs() <= tol,
							"strsm_right residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn trsm_bit_replay_both_sides() {
	let mut rng = Lcg(38);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- alpha * inv(A) * B, plain scal-then-strsv replay
				let acs = m + 1;
				let mut a = rng.mat_f32(m, m, acs);
				for j in 0..m {
					a[j * acs + j] = 2.0 * (m as f32) + 1.0 + j as f32;
				}
				let b0 = rng.mat_f32(m, n, bcs);
				let mut bt = b0.clone();
				strsm_left(0.9, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				for j in 0..n {
					for i in 0..m {
						br[j * bcs + i] *= 0.9;
					}
					if upper {
						for l in (0..m).rev() {
							if !unit {
								br[j * bcs + l] /= a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
						}
					} else {
						for l in 0..m {
							if !unit {
								br[j * bcs + l] /= a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
						}
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "strsm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- alpha * B * inv(A)
				let acs = n + 1;
				let mut a = rng.mat_f32(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = 2.0 * (n as f32) + 1.0 + j as f32;
				}
				let b0 = rng.mat_f32(m, n, bcs);
				let mut bt = b0.clone();
				strsm_right(0.9, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				let elim = |br: &mut Vec<f32>, j: usize, k: usize| {
					let t = -a[j * acs + k];
					for i in 0..m {
						br[j * bcs + i] += br[k * bcs + i] * t;
					}
				};
				let finish = |br: &mut Vec<f32>, j: usize| {
					if !unit {
						let s = 1.0 / a[j * acs + j];
						for i in 0..m {
							br[j * bcs + i] *= s;
						}
					}
				};
				if upper {
					// plain ascending replay — the grouping keeps upper
					// fully ascending, so this asserts bit-identity to it
					for j in 0..n {
						for i in 0..m {
							br[j * bcs + i] *= 0.9;
						}
						for k in 0..j {
							elim(&mut br, j, k);
						}
						finish(&mut br, j);
					}
				} else {
					// grouped replay: out-of-group solved columns first,
					// then in-group elimination descending
					let r = n % 4;
					let mut gs = n;
					while gs >= r + 4 {
						gs -= 4;
						for u in 0..4 {
							for i in 0..m {
								br[(gs + u) * bcs + i] *= 0.9;
							}
						}
						for k in gs + 4..n {
							for u in 0..4 {
								elim(&mut br, gs + u, k);
							}
						}
						for tc in (gs..gs + 4).rev() {
							for k in tc + 1..gs + 4 {
								elim(&mut br, tc, k);
							}
							finish(&mut br, tc);
						}
					}
					for j in (0..r).rev() {
						for i in 0..m {
							br[j * bcs + i] *= 0.9;
						}
						for k in j + 1..n {
							elim(&mut br, j, k);
						}
						finish(&mut br, j);
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "strsm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}
