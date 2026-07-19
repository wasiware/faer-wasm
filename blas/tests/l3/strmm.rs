use super::common::*;
use faer_wasm_blas::l3::*;

#[test]
fn trmm_both_sides_bounded() {
	let mut rng = Lcg(34);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				let a = rng.mat_f32(n, n, acs);
				let bcs = m + 2;
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

				let mut b = b0.clone();
				strmm_left(0.9, m, n, &a, acs, upper, unit, &mut b, bcs);
				for j in 0..n {
					for i in 0..m {
						let want = 0.9 * comp_sum32((0..n).map(|l| tri_at(i, l) * b0[j * bcs + l]));
						let scale = comp_sum32((0..n).map(|l| (tri_at(i, l) * b0[j * bcs + l]).abs()));
						let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 8.0 * scale + 1e-40;
						assert!(
							(b[j * bcs + i] as f64 - want).abs() <= tol,
							"strmm_left upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				let mut b = b0.clone();
				strmm_right(0.9, m, n, &a, acs, upper, unit, &mut b, bcs);
				for j in 0..n {
					for i in 0..m {
						let want = 0.9 * comp_sum32((0..n).map(|l| b0[l * bcs + i] * tri_at(l, j)));
						let scale = comp_sum32((0..n).map(|l| (b0[l * bcs + i] * tri_at(l, j)).abs()));
						let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 8.0 * scale + 1e-40;
						assert!(
							(b[j * bcs + i] as f64 - want).abs() <= tol,
							"strmm_right upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn trmm_bit_replay_both_sides() {
	let mut rng = Lcg(37);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- alpha * A * B, plain strmv-per-column replay
				let acs = m + 1;
				let a = rng.mat_f32(m, m, acs);
				let b0 = rng.mat_f32(m, n, bcs);
				let mut bt = b0.clone();
				strmm_left(0.9, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				for j in 0..n {
					if upper {
						for l in 0..m {
							let t = br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					} else {
						for l in (0..m).rev() {
							let t = br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					}
					for i in 0..m {
						br[j * bcs + i] *= 0.9;
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "strmm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- alpha * B * A
				let acs = n + 1;
				let a = rng.mat_f32(n, n, acs);
				let b0 = rng.mat_f32(m, n, bcs);
				let mut bt = b0.clone();
				strmm_right(0.9, m, n, &a, acs, upper, unit, &mut bt, bcs);
				let mut br = b0.clone();
				let dcol = |br: &mut Vec<f32>, j: usize, lo: usize, hi: usize| {
					let d = if unit { 1.0 } else { a[j * acs + j] };
					let s = 0.9 * d;
					for i in 0..m {
						br[j * bcs + i] *= s;
					}
					for k in lo..hi {
						let t = 0.9 * a[j * acs + k];
						for i in 0..m {
							br[j * bcs + i] += br[k * bcs + i] * t;
						}
					}
				};
				if upper {
					// grouped replay: in-group (descending targets, ascending
					// k) first, then out-of-group sources ascending
					let r = n % 4;
					let mut gs = n;
					while gs >= r + 4 {
						gs -= 4;
						for tc in (gs..gs + 4).rev() {
							dcol(&mut br, tc, gs, tc);
						}
						for k in 0..gs {
							for u in 0..4 {
								let t = 0.9 * a[(gs + u) * acs + k];
								for i in 0..m {
									br[(gs + u) * bcs + i] += br[k * bcs + i] * t;
								}
							}
						}
					}
					for j in (0..r).rev() {
						dcol(&mut br, j, 0, j);
					}
				} else {
					// plain ascending replay — the grouping keeps lower
					// fully ascending, so this asserts bit-identity to it
					for j in 0..n {
						dcol(&mut br, j, j + 1, n);
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "strmm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}
