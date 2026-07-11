//! Correctness gate for the fix-3 eigenvalue iteration: kernel Hessenberg +
//! hqr_eigvals must reproduce faer's eigenvalues (sorted complex compare)
//! across sizes, including above/below all routing thresholds, plus
//! conjugate-pair adjacency and the trace invariant.

use faer::prelude::*;
use faer::Mat;
use faer_wasm_kernels::hessenberg::hessenberg_factor_in_place;
use faer_wasm_kernels::schur_small::hqr_eigvals_in_place;

fn fill(n: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(n, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

#[test]
fn hqr_eigvals_match_faer() {
    for &n in &[1usize, 2, 3, 4, 5, 8, 16, 31, 32, 64, 96, 128, 192, 256] {
        let a = fill(n, 0xA24BAED4963EE407 ^ (n as u64));

        // kernel pipeline: hessenberg -> hqr (eigenvalues only)
        let mut h = a.clone();
        let mut tau = vec![0.0f64; n.saturating_sub(2).max(1)];
        let mut work = vec![0.0f64; n];
        hessenberg_factor_in_place(h.as_mut(), &mut tau, &mut work);
        for j in 0..n {
            for i in j + 2..n {
                h[(i, j)] = 0.0;
            }
        }
        let mut w_re = vec![0.0f64; n];
        let mut w_im = vec![0.0f64; n];
        let info = hqr_eigvals_in_place(h.as_mut(), &mut w_re, &mut w_im);
        assert!(info == 0, "n={n}: hqr did not converge (info={info})");

        // conjugate pairs adjacent, imaginary parts cancel
        let mut k = 0;
        while k < n {
            if w_im[k] != 0.0 {
                assert!(
                    k + 1 < n && w_im[k + 1] == -w_im[k] && w_re[k + 1] == w_re[k],
                    "n={n}: conjugate pair broken at {k}"
                );
                k += 2;
            } else {
                k += 1;
            }
        }

        // trace invariant
        let tr: f64 = (0..n).map(|i| a[(i, i)]).sum();
        let ws: f64 = w_re.iter().sum();
        assert!(
            (tr - ws).abs() < 1e-10 * (n.max(1) as f64),
            "n={n}: trace {tr} vs eigsum {ws}"
        );

        // sorted complex compare vs faer
        let fe: Vec<faer::c64> = a.eigenvalues().unwrap();
        let mut mine: Vec<(f64, f64)> = (0..n).map(|i| (w_re[i], w_im[i])).collect();
        let mut faers: Vec<(f64, f64)> = fe.iter().map(|z| (z.re, z.im)).collect();
        let cmp = |a: &(f64, f64), b: &(f64, f64)| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
        };
        mine.sort_by(cmp);
        faers.sort_by(cmp);
        for i in 0..n {
            let d = ((mine[i].0 - faers[i].0).powi(2) + (mine[i].1 - faers[i].1).powi(2)).sqrt();
            assert!(
                d < 1e-8 * (n.max(1) as f64),
                "n={n}: eigenvalue {i} differs from faer by {d:.2e} (mine {:?} vs faer {:?})",
                mine[i],
                faers[i]
            );
        }
    }
}
