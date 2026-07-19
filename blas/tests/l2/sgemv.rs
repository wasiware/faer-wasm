use super::common::*;
use faer_wasm_blas::l2::*;

#[test]
fn gemv_bit_for_bit_and_bounded() {
	let mut rng = Lcg(21);
	for &(m, n) in SHAPES {
		for &pad in PADS {
			let cs = m + pad;
			let a = rng.mat_f32(m, n, cs);
			let x = rng.vec_f32(n);
			let y0 = rng.vec_f32(m);
			for (alpha, beta) in [(1.0, 0.0), (-0.5, 1.0), (0.3, -2.0), (0.0, 0.5)] {
				let mut y = y0.clone();
				sgemv(alpha, m, n, &a, cs, &x, beta, &mut y);
				// same-order scalar reference: exact replay of column-saxpy
				let mut yr = y0.clone();
				if beta == 0.0 {
					yr.fill(0.0);
				} else if beta != 1.0 {
					for v in yr.iter_mut() {
						*v *= beta;
					}
				}
				for j in 0..n {
					let t = alpha * x[j];
					for i in 0..m {
						yr[i] += a[j * cs + i] * t;
					}
				}
				for i in 0..m {
					assert_eq!(y[i].to_bits(), yr[i].to_bits(), "sgemv bits {m}x{n} pad={pad} i={i}");
				}
				// independent bound: Kahan row-dots in a different order
				for i in 0..m {
					let want = comp_sum32((0..n).map(|j| alpha * x[j] * a[j * cs + i]))
						+ if beta == 0.0 { 0.0 } else { (beta * y0[i]) as f64 };
					let scale = comp_sum32((0..n).map(|j| (alpha * x[j] * a[j * cs + i]).abs()))
						+ ((beta * y0[i]).abs()) as f64;
					let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 4.0 * scale + 1e-40;
					assert!((y[i] as f64 - want).abs() <= tol, "sgemv bound {m}x{n} i={i}");
				}
			}
		}
	}
}

#[test]
fn gemv_t_bounded() {
	let mut rng = Lcg(22);
	for &(m, n) in SHAPES {
		let cs = m + 2;
		let a = rng.mat_f32(m, n, cs);
		let x = rng.vec_f32(m);
		let y0 = rng.vec_f32(n);
		let mut y = y0.clone();
		sgemv_t(0.7, m, n, &a, cs, &x, -0.3, &mut y);
		for j in 0..n {
			let want = 0.7 * comp_sum32((0..m).map(|i| a[j * cs + i] * x[i])) - 0.3 * y0[j] as f64;
			let scale = comp_sum32((0..m).map(|i| (a[j * cs + i] * x[i]).abs())) + y0[j].abs() as f64;
			let tol = f32::EPSILON as f64 * (m.max(1) as f64) * 4.0 * scale + 1e-40;
			assert!((y[j] as f64 - want).abs() <= tol, "sgemv_t {m}x{n} j={j}");
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn gemv_length_mismatch_panics() {
	sgemv(1.0, 2, 2, &[1.0, 2.0, 3.0, 4.0], 2, &[1.0], 0.0, &mut [0.0, 0.0]);
}

#[test]
#[should_panic(expected = "storage too short")]
fn gemv_short_storage_panics() {
	sgemv(1.0, 2, 2, &[1.0, 2.0, 3.0], 2, &[1.0, 1.0], 0.0, &mut [0.0, 0.0]);
}
