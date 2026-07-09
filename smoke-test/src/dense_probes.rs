//! Foundation correctness probes: the "simple" dense decompositions checked
//! at sizes that exercise the blocked/SIMD code paths (n = 96) and the SIMD
//! tail lanes (n = 33, odd), in both f64 and c64.
//!
//! Same design as `schur_probe`: each check is a residual/property test with
//! a tolerance ~2+ orders of magnitude above the observed error (see the
//! `margins` bin for the measured values), summed into an integer score that
//! is exact across native/wasm/simd128/relaxed-simd. The tiny 3×3 probes in
//! `lib.rs` pin exact bits on fixed values; these probes pin *correctness at
//! realistic sizes* — the regime where the pulp relaxed-simd c64 bug and
//! faer's below-subdiagonal workspace junk lived, invisible to small probes.
//!
//! Expected scores: `dense_f64_probe` = 26, `dense_c64_probe` = 24
//! (13/12 checks per size × 2 sizes).

use faer::prelude::*;
use faer::{Mat, Side};

// deterministic fill (same LCG as bench/), values in [-1, 1]
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn max_abs_diff(a: MatRef<f64>, b: MatRef<f64>) -> f64 {
    let mut e = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let d = a[(i, j)] - b[(i, j)];
            e = if d.abs() > e { d.abs() } else { e };
        }
    }
    e
}

fn max_off_identity(a: MatRef<f64>) -> f64 {
    let mut e = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let id = if i == j { 1.0 } else { 0.0 };
            let d = a[(i, j)] - id;
            e = if d.abs() > e { d.abs() } else { e };
        }
    }
    e
}

fn max_abs_diff_c(a: MatRef<c64>, b: MatRef<c64>) -> f64 {
    let mut e = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let d = a[(i, j)] - b[(i, j)];
            let m = d.re * d.re + d.im * d.im;
            e = if m > e { m } else { e };
        }
    }
    e
}

fn max_off_identity_c(a: MatRef<c64>) -> f64 {
    let mut e = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let id = if i == j { 1.0 } else { 0.0 };
            let d = a[(i, j)] - c64::new(id, 0.0);
            let m = d.re * d.re + d.im * d.im;
            e = if m > e { m } else { e };
        }
    }
    e
}

/// raw f64 error magnitudes at one size, in check order (used by the score
/// and by the `margins` calibration bin)
pub fn f64_errors(n: usize) -> [f64; 13] {
    let mut next = lcg(0x9E3779B97F4A7C15 ^ (n as u64));
    let a = Mat::from_fn(n, n, |_, _| next());
    let b = Mat::from_fn(n, 1, |_, _| next());

    // 0: LU solve residual ||Ax - b||
    let x = a.partial_piv_lu().solve(&b);
    let r = &a * &x - &b;
    let e_lu = (0..n).fold(0.0f64, |m, i| m.max(r[(i, 0)].abs()));

    // 1-2: QR reconstruction + orthogonality
    let qr = a.qr();
    let q = qr.compute_thin_Q();
    let e_qr = max_abs_diff((&q * qr.R()).as_ref(), a.as_ref());
    let e_qo = max_off_identity((q.transpose() * &q).as_ref());

    // SPD matrix for LLT: AAᵀ/n + I
    let spd = &a * a.transpose() * faer::Scale(1.0 / n as f64) + Mat::<f64>::identity(n, n);
    // 3-4: LLT reconstruction + solve residual
    let llt = spd.llt(Side::Lower).unwrap();
    let l = llt.L();
    let e_ll = max_abs_diff((l * l.transpose()).as_ref(), spd.as_ref());
    let xs = llt.solve(&b);
    let rs = &spd * &xs - &b;
    let e_ls = (0..n).fold(0.0f64, |m, i| m.max(rs[(i, 0)].abs()));

    // 5-8: SVD reconstruction, U/V orthogonality, singular values sorted >= 0
    let svd = a.svd().unwrap();
    let (u, v, s) = (svd.U(), svd.V(), svd.S());
    let smat = Mat::from_fn(n, n, |i, j| if i == j { s[i] } else { 0.0 });
    let e_sv = max_abs_diff((u * &smat * v.transpose()).as_ref(), a.as_ref());
    let e_su = max_off_identity((u.transpose() * u).as_ref());
    let e_svv = max_off_identity((v.transpose() * v).as_ref());
    let mut sorted = true;
    for k in 0..n {
        sorted &= s[k] >= 0.0 && (k + 1 >= n || s[k] >= s[k + 1]);
    }

    // symmetric matrix for the self-adjoint EVD
    let sym = &a + a.transpose();
    // 9-10: A U = U Λ residual + eigenvector orthogonality
    let evd = sym.self_adjoint_eigen(Side::Lower).unwrap();
    let (eu, es) = (evd.U(), evd.S());
    let lmat = Mat::from_fn(n, n, |i, j| if i == j { es[i] } else { 0.0 });
    let e_ev = max_abs_diff((&sym * eu).as_ref(), (eu * &lmat).as_ref());
    let e_eo = max_off_identity((eu.transpose() * eu).as_ref());

    // 11: general eigenvalues: sum == trace (order-independent)
    let ev: alloc::vec::Vec<c64> = a.eigenvalues().unwrap();
    let sum = ev.iter().fold(c64::new(0.0, 0.0), |acc, v| acc + *v);
    let tr: f64 = (0..n).map(|k| a[(k, k)]).sum();
    // squared norm: f64::hypot is std-only, unavailable in the no_std build
    let e_tr = (sum.re - tr) * (sum.re - tr) + sum.im * sum.im;

    // 12: self-adjoint eigenvalue sum == trace of sym
    let str_: f64 = (0..n).map(|k| sym[(k, k)]).sum();
    let ssum: f64 = (0..n).map(|k| es[k]).sum();
    let e_str = (str_ - ssum).abs();

    [
        e_lu,
        e_qr,
        e_qo,
        e_ll,
        e_ls,
        e_sv,
        e_su,
        e_svv,
        if sorted { 0.0 } else { 1.0 },
        e_ev,
        e_eo,
        e_tr,
        e_str,
    ]
}

/// raw c64 error magnitudes at one size (SQUARED norms for the matrix
/// residuals, matching `max_abs_diff_c`)
pub fn c64_errors(n: usize) -> [f64; 12] {
    let mut next = lcg(0xD1B54A32D192ED03 ^ (n as u64));
    let mut nx = move || {
        let re = next();
        let im = next();
        c64::new(re, im)
    };
    let a = Mat::from_fn(n, n, |_, _| nx());
    let b = Mat::from_fn(n, 1, |_, _| nx());

    // 0: LU solve residual
    let x = a.partial_piv_lu().solve(&b);
    let r = &a * &x - &b;
    let e_lu = (0..n).fold(0.0f64, |m, i| {
        let d = r[(i, 0)];
        m.max(d.re * d.re + d.im * d.im)
    });

    // 1-2: QR reconstruction + unitarity
    let qr = a.qr();
    let q = qr.compute_thin_Q();
    let e_qr = max_abs_diff_c((&q * qr.R()).as_ref(), a.as_ref());
    let e_qo = max_off_identity_c((q.adjoint() * &q).as_ref());

    // HPD matrix for LLT: AAᴴ/n + I
    let hpd = &a * a.adjoint() * faer::Scale(c64::new(1.0 / n as f64, 0.0))
        + Mat::<c64>::identity(n, n);
    // 3-4: LLT reconstruction + solve residual
    let llt = hpd.llt(Side::Lower).unwrap();
    let l = llt.L();
    let e_ll = max_abs_diff_c((l * l.adjoint()).as_ref(), hpd.as_ref());
    let xs = llt.solve(&b);
    let rs = &hpd * &xs - &b;
    let e_ls = (0..n).fold(0.0f64, |m, i| {
        let d = rs[(i, 0)];
        m.max(d.re * d.re + d.im * d.im)
    });

    // 5-8: SVD reconstruction, U/V unitarity, singular values sorted >= 0
    let svd = a.svd().unwrap();
    let (u, v, s) = (svd.U(), svd.V(), svd.S());
    let smat = Mat::from_fn(n, n, |i, j| if i == j { s[i] } else { c64::new(0.0, 0.0) });
    let e_sv = max_abs_diff_c((u * &smat * v.adjoint()).as_ref(), a.as_ref());
    let e_su = max_off_identity_c((u.adjoint() * u).as_ref());
    let e_svv = max_off_identity_c((v.adjoint() * v).as_ref());
    let mut sorted = true;
    for k in 0..n {
        // singular values of a complex matrix are real non-negative; faer
        // stores them as c64 with zero imaginary part
        sorted &= s[k].im == 0.0 && s[k].re >= 0.0 && (k + 1 >= n || s[k].re >= s[k + 1].re);
    }

    // Hermitian matrix for the self-adjoint EVD
    let herm = &a + a.adjoint();
    // 9-10: A U = U Λ residual + eigenvector unitarity
    let evd = herm.self_adjoint_eigen(Side::Lower).unwrap();
    let (eu, es) = (evd.U(), evd.S());
    let lmat = Mat::from_fn(n, n, |i, j| if i == j { es[i] } else { c64::new(0.0, 0.0) });
    let e_ev = max_abs_diff_c((&herm * eu).as_ref(), (eu * &lmat).as_ref());
    let e_eo = max_off_identity_c((eu.adjoint() * eu).as_ref());

    // 11: general eigenvalues: sum == trace (this exact call was garbage
    // under relaxed-simd before patches/pulp/0003)
    let ev: alloc::vec::Vec<c64> = a.eigenvalues().unwrap();
    let sum = ev.iter().fold(c64::new(0.0, 0.0), |acc, v| acc + *v);
    let tr = (0..n).fold(c64::new(0.0, 0.0), |acc, k| acc + a[(k, k)]);
    let d = sum - tr;
    let e_tr = d.re * d.re + d.im * d.im;

    [
        e_lu, e_qr, e_qo, e_ll, e_ls, e_sv, e_su, e_svv,
        if sorted { 0.0 } else { 1.0 },
        e_ev, e_eo, e_tr,
    ]
}

// Tolerances, one per check, calibrated from the `margins` bin output
// (native, 2026-07-09): each sits >= 2 orders of magnitude above the largest
// observed error across n=33/96 and f64/c64 where applicable. c64 residual
// tolerances are for SQUARED norms.
const F64_TOL: [f64; 13] = [
    1e-10, // lu solve
    1e-12, // qr recon
    1e-12, // qr orth
    1e-12, // llt recon
    1e-11, // llt solve
    1e-12, // svd recon
    1e-12, // svd U orth
    1e-12, // svd V orth
    0.5,   // singular values sorted (boolean)
    1e-11, // sa-evd residual
    1e-12, // sa-evd orth
    1e-20, // eigenvalue trace (squared)
    1e-10, // sa eigenvalue trace
];
const C64_TOL: [f64; 12] = [
    1e-20, // lu solve (squared)
    1e-24, // qr recon (squared)
    1e-24, // qr orth (squared)
    1e-24, // llt recon (squared)
    1e-22, // llt solve (squared)
    1e-24, // svd recon (squared)
    1e-24, // svd U orth (squared)
    1e-24, // svd V orth (squared)
    0.5,   // singular values real, sorted (boolean)
    1e-22, // evd residual (squared)
    1e-24, // evd orth (squared)
    1e-20, // eigenvalue trace (squared)
];

pub fn f64_score() -> f64 {
    let mut score = 0.0;
    for n in [33usize, 96] {
        let e = f64_errors(n);
        for k in 0..13 {
            if e[k] < F64_TOL[k] {
                score += 1.0;
            }
        }
    }
    score
}

pub fn c64_score() -> f64 {
    let mut score = 0.0;
    for n in [33usize, 96] {
        let e = c64_errors(n);
        for k in 0..12 {
            if e[k] < C64_TOL[k] {
                score += 1.0;
            }
        }
    }
    score
}
