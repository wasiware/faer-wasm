use super::common::*;
use faer_wasm_blas::l2::*;

#[test]
fn symv_bounded_both_triangles() {
	let mut rng = Lcg(24);
	for &n in NS {
		for upper in [true, false] {
			let cs = n + 1;
			// build a full symmetric matrix, then expose only one triangle
			let full = {
				let mut f = vec![0.0; n * n];
				for j in 0..n {
					for i in 0..=j {
						let v = rng.next_f64();
						f[j * n + i] = v;
						f[i * n + j] = v;
					}
				}
				f
			};
			let mut a = vec![f64::NAN; if n == 0 { 0 } else { cs * (n - 1) + n }];
			for j in 0..n {
				for i in 0..n {
					let stored = if upper { i <= j } else { i >= j };
					if stored {
						a[j * cs + i] = full[j * n + i];
					}
				}
			}
			let x = rng.vec_f64(n);
			let y0 = rng.vec_f64(n);
			let mut y = y0.clone();
			dsymv(0.9, n, &a, cs, upper, &x, 0.4, &mut y);
			for i in 0..n {
				let want =
					0.9 * comp_sum((0..n).map(|j| full[j * n + i] * x[j])) + 0.4 * y0[i];
				let scale =
					comp_sum((0..n).map(|j| (full[j * n + i] * x[j]).abs())) + y0[i].abs();
				let tol = f64::EPSILON * (n.max(1) as f64) * 4.0 * scale + 1e-300;
				assert!((y[i] - want).abs() <= tol, "dsymv upper={upper} n={n} i={i}");
			}
		}
	}
}
