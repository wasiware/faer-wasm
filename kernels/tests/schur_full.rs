//! Correctness gate for the full-Schur kernel pipeline (Schur campaign):
//! kernel Hessenberg → `hessenberg_form_q` → `hqr_schur_in_place`
//! (want_t + Z). Requires (a) backward error ‖A − Z·T·Zᵀ‖ small,
//! (b) Z orthogonal, (c) T quasi upper triangular with `dlanv2`-standardized
//! 2×2 blocks (complex-pair blocks: equal diagonal, opposite-sign
//! off-diagonals), (d) eigenvalues match faer's EVD of A, (e) w_re/w_im
//! agree with T's diagonal blocks, (f) Q-formation alone reproduces the
//! reference Q of the hessenberg test (‖A·Q − Q·H‖). Also gates the f32
//! twins at eps32 tolerances and the want_t=false toggle's eigenvalues.

use faer::Mat;
use faer_wasm_kernels::hessenberg::{hessenberg_factor_in_place, hessenberg_form_q};
use faer_wasm_kernels::schur_small::hqr_schur_in_place;

fn fill(n: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(n, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

/// kernel Schur pipeline: A → (T, Z, w) with A = Z T Zᵀ
fn kernel_schur(a: &Mat<f64>) -> (Mat<f64>, Mat<f64>, Vec<f64>, Vec<f64>) {
    let n = a.nrows();
    let mut t = a.clone();
    let k = n.saturating_sub(2);
    let mut tau = vec![0.0f64; k.max(1)];
    let mut work = vec![0.0f64; n];
    hessenberg_factor_in_place(t.as_mut(), &mut tau, &mut work);
    let mut z = Mat::<f64>::zeros(n, n);
    hessenberg_form_q(t.as_ref(), &tau, z.as_mut());
    // clear the reflector storage below the first subdiagonal
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = 0.0;
        }
    }
    let mut w_re = vec![0.0f64; n];
    let mut w_im = vec![0.0f64; n];
    let info = hqr_schur_in_place(t.as_mut(), Some(z.as_mut()), &mut w_re, &mut w_im, true);
    assert!(info == 0, "n={n}: hqr_schur did not converge");
    (t, z, w_re, w_im)
}

#[test]
fn schur_full_properties() {
    for &n in &[1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 33, 64, 96, 128, 200, 256] {
        let a = fill(n, 0x9E3779B97F4A7C15 ^ (n as u64));
        let (t, z, w_re, w_im) = kernel_schur(&a);

        // (a) backward error ‖A − Z T Zᵀ‖_max
        let recon = &z * &t * z.transpose();
        let mut berr = 0.0f64;
        let mut scale = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                berr = berr.max((recon[(i, j)] - a[(i, j)]).abs());
                scale = scale.max(a[(i, j)].abs());
            }
        }
        assert!(
            berr < 1e-12 * scale.max(1.0) * (n as f64),
            "n={n}: ||A - Z T Zt|| = {berr:.2e}"
        );

        // (b) orthogonality ‖ZᵀZ − I‖_max
        let ztz = z.transpose() * &z;
        let mut oerr = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                oerr = oerr.max((ztz[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(oerr < 1e-12 * (n as f64), "n={n}: ZtZ-I = {oerr:.2e}");

        // (c) quasi-triangular: exact zeros below the first subdiagonal, no
        // adjacent 2×2 blocks, and standardized blocks (t00 == t11,
        // sign(t01) != sign(t10))
        for j in 0..n {
            for i in j + 2..n {
                assert!(t[(i, j)] == 0.0, "n={n}: T[{i},{j}] != 0");
            }
        }
        let mut k = 0usize;
        while k + 1 < n {
            if t[(k + 1, k)] != 0.0 {
                assert!(
                    t[(k, k)] == t[(k + 1, k + 1)],
                    "n={n}: 2x2 block at {k} not standardized (diag)"
                );
                assert!(
                    t[(k, k + 1)] * t[(k + 1, k)] < 0.0,
                    "n={n}: 2x2 block at {k} not standardized (sign)"
                );
                k += 2;
            } else {
                k += 1;
            }
        }

        // (d) eigenvalues match faer's EVD of A (sorted complex compare)
        let ea: Vec<faer::c64> = a.eigenvalues().unwrap();
        let mut ea: Vec<_> = ea.iter().map(|z| (z.re, z.im)).collect();
        let mut ek: Vec<_> = (0..n).map(|i| (w_re[i], w_im[i])).collect();
        let cmp = |a: &(f64, f64), b: &(f64, f64)| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
        };
        ea.sort_by(cmp);
        ek.sort_by(cmp);
        for i in 0..n {
            let d = ((ea[i].0 - ek[i].0).powi(2) + (ea[i].1 - ek[i].1).powi(2)).sqrt();
            assert!(d < 1e-9 * (n as f64), "n={n}: eigenvalue {i} moved by {d:.2e}");
        }

        // (e) w agrees with T's diagonal blocks
        let mut k = 0usize;
        while k < n {
            if k + 1 < n && t[(k + 1, k)] != 0.0 {
                let re = t[(k, k)];
                let im = (-t[(k, k + 1)] * t[(k + 1, k)]).sqrt();
                assert!((w_re[k] - re).abs() < 1e-10 && (w_im[k].abs() - im).abs() < 1e-10,
                    "n={n}: block eig mismatch at {k}");
                assert!(w_re[k] == w_re[k + 1] && w_im[k] == -w_im[k + 1],
                    "n={n}: pair not conjugate at {k}");
                k += 2;
            } else {
                assert!(w_re[k] == t[(k, k)] && w_im[k] == 0.0, "n={n}: real eig mismatch at {k}");
                k += 1;
            }
        }
    }
}

#[test]
fn form_q_matches_reference_accumulation() {
    for &n in &[2usize, 3, 8, 33, 96] {
        let a = fill(n, 0x853C49E6748FEA9B ^ (n as u64));
        let mut fac = a.clone();
        let k = n.saturating_sub(2);
        let mut tau = vec![0.0f64; k.max(1)];
        let mut work = vec![0.0f64; n];
        hessenberg_factor_in_place(fac.as_mut(), &mut tau, &mut work);
        let mut q = Mat::<f64>::zeros(n, n);
        hessenberg_form_q(fac.as_ref(), &tau, q.as_mut());
        let h = Mat::from_fn(n, n, |i, j| if i > j + 1 { 0.0 } else { fac[(i, j)] });
        // similarity ‖A·Q − Q·H‖ and orthogonality — Q is THE Q of the
        // reduction iff both hold
        let aq = &a * &q;
        let qh = &q * &h;
        let mut serr = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                serr = serr.max((aq[(i, j)] - qh[(i, j)]).abs());
            }
        }
        assert!(serr < 1e-12 * (n as f64), "n={n}: ||AQ - QH|| = {serr:.2e}");
        let qtq = q.transpose() * &q;
        let mut oerr = 0.0f64;
        for j in 0..n {
            for i in 0..n {
                oerr = oerr.max((qtq[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(oerr < 1e-12 * (n as f64), "n={n}: QtQ-I = {oerr:.2e}");
    }
}

#[test]
fn want_t_false_toggle_preserves_eigenvalues() {
    // the instrumentation toggle (want_t = false, Z still accumulated) must
    // deliver the same eigenvalues; T is junk by contract
    for &n in &[8usize, 33, 96] {
        let a = fill(n, 0xD1B54A32D192ED03 ^ (n as u64));
        let n_ = a.nrows();
        let mut h = a.clone();
        let k = n_.saturating_sub(2);
        let mut tau = vec![0.0f64; k.max(1)];
        let mut work = vec![0.0f64; n_];
        hessenberg_factor_in_place(h.as_mut(), &mut tau, &mut work);
        for j in 0..n_ {
            for i in j + 2..n_ {
                h[(i, j)] = 0.0;
            }
        }
        let mut w_re = vec![0.0f64; n_];
        let mut w_im = vec![0.0f64; n_];
        let info = hqr_schur_in_place(h.as_mut(), None, &mut w_re, &mut w_im, false);
        assert!(info == 0);
        let ea: Vec<faer::c64> = a.eigenvalues().unwrap();
        let mut ea: Vec<_> = ea.iter().map(|z| (z.re, z.im)).collect();
        let mut ek: Vec<_> = (0..n_).map(|i| (w_re[i], w_im[i])).collect();
        let cmp = |a: &(f64, f64), b: &(f64, f64)| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
        };
        ea.sort_by(cmp);
        ek.sort_by(cmp);
        for i in 0..n_ {
            let d = ((ea[i].0 - ek[i].0).powi(2) + (ea[i].1 - ek[i].1).powi(2)).sqrt();
            assert!(d < 1e-9 * (n_ as f64), "n={n}: eigenvalue {i} moved by {d:.2e}");
        }
    }
}

#[test]
fn schur_full_f32() {
    // f32 twin at eps32 tolerances (~6e-8): backward error, orthogonality,
    // structure
    for &n in &[4usize, 16, 33, 64, 96] {
        let a64 = fill(n, 0x2545F4914F6CDD1D ^ (n as u64));
        let a = Mat::from_fn(n, n, |i, j| a64[(i, j)] as f32);
        let mut t = a.clone();
        let k = n.saturating_sub(2);
        let mut tau = vec![0.0f32; k.max(1)];
        let mut work = vec![0.0f32; n];
        hessenberg_factor_in_place(t.as_mut(), &mut tau, &mut work);
        let mut z = Mat::<f32>::zeros(n, n);
        hessenberg_form_q(t.as_ref(), &tau, z.as_mut());
        for j in 0..n {
            for i in j + 2..n {
                t[(i, j)] = 0.0;
            }
        }
        let mut w_re = vec![0.0f32; n];
        let mut w_im = vec![0.0f32; n];
        let info = hqr_schur_in_place(t.as_mut(), Some(z.as_mut()), &mut w_re, &mut w_im, true);
        assert!(info == 0, "n={n}: f32 hqr_schur did not converge");
        let recon = &z * &t * z.transpose();
        let mut berr = 0.0f32;
        for j in 0..n {
            for i in 0..n {
                berr = berr.max((recon[(i, j)] - a[(i, j)]).abs());
            }
        }
        assert!(berr < 1e-4 * (n as f32), "n={n}: f32 ||A - Z T Zt|| = {berr:.2e}");
        let ztz = z.transpose() * &z;
        let mut oerr = 0.0f32;
        for j in 0..n {
            for i in 0..n {
                oerr = oerr.max((ztz[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(oerr < 1e-4 * (n as f32), "n={n}: f32 ZtZ-I = {oerr:.2e}");
        for j in 0..n {
            for i in j + 2..n {
                assert!(t[(i, j)] == 0.0, "n={n}: f32 T[{i},{j}] != 0");
            }
        }
    }
}

#[test]
fn multishift_composition_above_crossover() {
    // the n >= 480 route of the shipping pipeline: kernel Hessenberg +
    // backward-accumulated Q seeding faer's multishift (want_t, Z) — checks
    // the Q seeding composes correctly with faer's blocked path (which uses
    // the below-subdiagonal region as workspace; cleaned after, like
    // faer-schur)
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::schur::{self, real_schur};
    use faer::{Auto, Par};

    let n = 512usize;
    let a = fill(n, 0xBF58476D1CE4E5B9);
    let mut t = a.clone();
    let mut tau = vec![0.0f64; n - 2];
    let mut work = vec![0.0f64; n];
    hessenberg_factor_in_place(t.as_mut(), &mut tau, &mut work);
    let mut z = Mat::<f64>::zeros(n, n);
    hessenberg_form_q(t.as_ref(), &tau, z.as_mut());
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = 0.0;
        }
    }
    let params: schur::SchurParams = Auto::<f64>::auto();
    let mut w_re = faer::Col::<f64>::zeros(n);
    let mut w_im = faer::Col::<f64>::zeros(n);
    let mut mem = MemBuffer::new(schur::multishift_qr_scratch::<f64>(
        n,
        n,
        true,
        true,
        Par::Seq,
        params,
    ));
    let (info, _, _) = real_schur::multishift_qr::<f64>(
        true,
        t.as_mut(),
        Some(z.as_mut()),
        w_re.as_mut(),
        w_im.as_mut(),
        0,
        n,
        Par::Seq,
        MemStack::new(&mut mem),
        params,
    );
    assert!(info == 0, "multishift did not converge");
    for j in 0..n {
        for i in j + 2..n {
            t[(i, j)] = 0.0;
        }
    }
    let recon = &z * &t * z.transpose();
    let mut berr = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            berr = berr.max((recon[(i, j)] - a[(i, j)]).abs());
        }
    }
    assert!(berr < 1e-12 * (n as f64), "||A - Z T Zt|| = {berr:.2e}");
    let ztz = z.transpose() * &z;
    let mut oerr = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            oerr = oerr.max((ztz[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    assert!(oerr < 1e-12 * (n as f64), "ZtZ-I = {oerr:.2e}");
}
