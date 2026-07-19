use super::common::*;
use faer_wasm_blas::l2::*;

#[test]
fn gemv_bit_for_bit_and_bounded() {
	let mut rng = Lcg(21);
	for &(m, n) in SHAPES {
		for &pad in PADS {
			let cs = m + pad;
			let a = rng.mat_f64(m, n, cs);
			let x = rng.vec_f64(n);
			let y0 = rng.vec_f64(m);
			for (alpha, beta) in [(1.0, 0.0), (-0.5, 1.0), (0.3, -2.0), (0.0, 0.5)] {
				let mut y = y0.clone();
				dgemv(alpha, m, n, &a, cs, &x, beta, &mut y);
				// same-order scalar reference: exact replay of column-daxpy
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
					assert_eq!(y[i].to_bits(), yr[i].to_bits(), "dgemv bits {m}x{n} pad={pad} i={i}");
				}
				// independent bound: Kahan row-dots in a different order
				for i in 0..m {
					let want = comp_sum((0..n).map(|j| alpha * x[j] * a[j * cs + i]))
						+ if beta == 0.0 { 0.0 } else { beta * y0[i] };
					let scale = comp_sum((0..n).map(|j| (alpha * x[j] * a[j * cs + i]).abs()))
						+ (beta * y0[i]).abs();
					let tol = f64::EPSILON * (n.max(1) as f64) * 4.0 * scale + 1e-300;
					assert!((y[i] - want).abs() <= tol, "dgemv bound {m}x{n} i={i}");
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
		let a = rng.mat_f64(m, n, cs);
		let x = rng.vec_f64(m);
		let y0 = rng.vec_f64(n);
		let mut y = y0.clone();
		dgemv_t(0.7, m, n, &a, cs, &x, -0.3, &mut y);
		for j in 0..n {
			let want = 0.7 * comp_sum((0..m).map(|i| a[j * cs + i] * x[i])) - 0.3 * y0[j];
			let scale = comp_sum((0..m).map(|i| (a[j * cs + i] * x[i]).abs())) + y0[j].abs();
			let tol = f64::EPSILON * (m.max(1) as f64) * 4.0 * scale + 1e-300;
			assert!((y[j] - want).abs() <= tol, "dgemv_t {m}x{n} j={j}");
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn gemv_length_mismatch_panics() {
	dgemv(1.0, 2, 2, &[1.0, 2.0, 3.0, 4.0], 2, &[1.0], 0.0, &mut [0.0, 0.0]);
}

#[test]
#[should_panic(expected = "storage too short")]
fn gemv_short_storage_panics() {
	dgemv(1.0, 2, 2, &[1.0, 2.0, 3.0], 2, &[1.0, 1.0], 0.0, &mut [0.0, 0.0]);
}
