//! Correctness gate for the c64 Schur kernel pipeline: complex Hessenberg
//! → backward-accumulated Q → single-shift chqr (want_t + Z). Requires
//! (a) backward error ‖A − Z·T·Zᴴ‖ small, (b) Z unitary, (c) T strictly
//! upper triangular with exact subdiagonal zeros, (d) eigenvalues match
//! faer's EVD of A, (e) diag(T) == w, plus a Hessenberg-only similarity/
//! unitarity check and the want_t=false toggle.

use faer::{c64, Mat};
use faer_wasm_kernels::hessenberg_cplx::{hessenberg_cplx_factor_in_place, hessenberg_cplx_form_q};
use faer_wasm_kernels::schur_small_cplx::chqr_schur_in_place;

fn fill_c(n: usize, mut s: u64) -> Mat<c64> {
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    Mat::from_fn(n, n, |_, _| {
        let re = next();
        let im = next();
        c64::new(re, im)
    })
}

fn cmod(z: c64) -> f64 {
    (z.re * z.re + z.im * z.im).sqrt()
}

/// kernel c64 Schur pipeline: A → (T, Z, w) with A = Z T Zᴴ
fn kernel_schur_c64(a: &Mat<c64>) -> (Mat<c64>, Mat<c64>, Vec<c64>) {
    let n = a.nrows();
    let mut t = a.clone();
    let k = n.saturating_sub(2);
    let mut tau = vec![c64::new(0.0, 0.0); k.max(1)];
    let mut work = vec![c64::new(0.0, 0.0); n];
    hessenberg_cplx_factor_in_place(t.as_mut(), &mut tau, &mut work);
    let mut z = Mat::<c64>::zeros(n, n);
    hessenberg_cplx_form_q(t.as_ref(), &tau, z.as_mut());
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = c64::new(0.0, 0.0);
        }
    }
    let mut w = vec![c64::new(0.0, 0.0); n];
    let info = chqr_schur_in_place(t.as_mut(), Some(z.as_mut()), &mut w, true);
    assert!(info == 0, "n={n}: chqr_schur did not converge");
    (t, z, w)
}

#[test]
fn hessenberg_cplx_similarity_unitarity() {
    for &n in &[2usize, 3, 8, 33, 96] {
        let a = fill_c(n, 0x853C49E6748FEA9B ^ (n as u64));
        let mut fac = a.clone();
        let k = n.saturating_sub(2);
        let mut tau = vec![c64::new(0.0, 0.0); k.max(1)];
        let mut work = vec![c64::new(0.0, 0.0); n];
        hessenberg_cplx_factor_in_place(fac.as_mut(), &mut tau, &mut work);
        let mut q = Mat::<c64>::zeros(n, n);
        hessenberg_cplx_form_q(fac.as_ref(), &tau, q.as_mut());
        let h = Mat::from_fn(n, n, |i, j| {
            if i > j + 1 {
                c64::new(0.0, 0.0)
            } else {
                fac[(i, j)]
            }
        });
        let aq = &a * &q;
        let qh = &q * &h;
        let mut serr = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                serr = serr.max(cmod(aq[(i, j)] - qh[(i, j)]));
            }
        }
        assert!(serr < 1e-12 * (n as f64), "n={n}: ||AQ - QH|| = {serr:.2e}");
        let qhq = q.adjoint() * &q;
        let mut oerr = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                let id = if i == j { 1.0 } else { 0.0 };
                oerr = oerr.max(cmod(qhq[(i, j)] - c64::new(id, 0.0)));
            }
        }
        assert!(oerr < 1e-12 * (n as f64), "n={n}: QhQ-I = {oerr:.2e}");
    }
}

#[test]
fn schur_cplx_full_properties() {
    for &n in &[1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 33, 64, 96, 128, 200, 256] {
        let a = fill_c(n, 0x9E3779B97F4A7C15 ^ (n as u64));
        let (t, z, w) = kernel_schur_c64(&a);

        // (a) backward error ‖A − Z T Zᴴ‖_max
        let recon = &z * &t * z.adjoint();
        let mut berr = 0.0f64;
        let mut scale = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                berr = berr.max(cmod(recon[(i, j)] - a[(i, j)]));
                scale = scale.max(cmod(a[(i, j)]));
            }
        }
        assert!(
            berr < 1e-12 * scale.max(1.0) * (n as f64),
            "n={n}: ||A - Z T Zh|| = {berr:.2e}"
        );

        // (b) unitarity ‖ZᴴZ − I‖_max
        let zhz = z.adjoint() * &z;
        let mut oerr = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                let id = if i == j { 1.0 } else { 0.0 };
                oerr = oerr.max(cmod(zhz[(i, j)] - c64::new(id, 0.0)));
            }
        }
        assert!(oerr < 1e-12 * (n as f64), "n={n}: ZhZ-I = {oerr:.2e}");

        // (c) strictly upper triangular with exact zeros below the diagonal
        for j in 0..n {
            for i in j + 1..n {
                assert!(
                    t[(i, j)] == c64::new(0.0, 0.0),
                    "n={n}: T[{i},{j}] != 0"
                );
            }
        }

        // (d) eigenvalues match faer's EVD of A (sorted complex compare)
        let ea: Vec<c64> = a.eigenvalues().unwrap();
        let mut ea: Vec<_> = ea.iter().map(|z| (z.re, z.im)).collect();
        let mut ek: Vec<_> = w.iter().map(|z| (z.re, z.im)).collect();
        let cmp = |a: &(f64, f64), b: &(f64, f64)| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
        };
        ea.sort_by(cmp);
        ek.sort_by(cmp);
        for i in 0..n {
            let d = ((ea[i].0 - ek[i].0).powi(2) + (ea[i].1 - ek[i].1).powi(2)).sqrt();
            assert!(d < 1e-9 * (n as f64), "n={n}: eigenvalue {i} moved by {d:.2e}");
        }

        // (e) diag(T) == w exactly
        for i in 0..n {
            assert!(w[i] == t[(i, i)], "n={n}: w[{i}] != T[{i},{i}]");
        }
    }
}

#[test]
fn chqr_want_t_false_toggle_preserves_eigenvalues() {
    for &n in &[8usize, 33, 96] {
        let a = fill_c(n, 0xD1B54A32D192ED03 ^ (n as u64));
        let mut h = a.clone();
        let k = n.saturating_sub(2);
        let mut tau = vec![c64::new(0.0, 0.0); k.max(1)];
        let mut work = vec![c64::new(0.0, 0.0); n];
        hessenberg_cplx_factor_in_place(h.as_mut(), &mut tau, &mut work);
        for j in 0..n {
            for i in j + 2..n {
                h[(i, j)] = c64::new(0.0, 0.0);
            }
        }
        let mut w = vec![c64::new(0.0, 0.0); n];
        let info = chqr_schur_in_place(h.as_mut(), None, &mut w, false);
        assert!(info == 0);
        let ea: Vec<c64> = a.eigenvalues().unwrap();
        let mut ea: Vec<_> = ea.iter().map(|z| (z.re, z.im)).collect();
        let mut ek: Vec<_> = w.iter().map(|z| (z.re, z.im)).collect();
        let cmp = |a: &(f64, f64), b: &(f64, f64)| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
        };
        ea.sort_by(cmp);
        ek.sort_by(cmp);
        for i in 0..n {
            let d = ((ea[i].0 - ek[i].0).powi(2) + (ea[i].1 - ek[i].1).powi(2)).sqrt();
            assert!(d < 1e-9 * (n as f64), "n={n}: eigenvalue {i} moved by {d:.2e}");
        }
    }
}
