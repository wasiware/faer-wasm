//! Property gate for the wasm-shaped unblocked Householder QR:
//! backward error ‖A − Q·R‖ (Q reconstructed from the stored reflectors),
//! orthogonality of the reconstructed Q, and agreement of |R| with faer's
//! own QR — across odd sizes, SIMD-tail sizes, and tall/wide/square shapes.

use faer::prelude::*;
use faer::Mat;
use faer_wasm_kernels::qr::qr_factor_in_place;

fn fill(m: usize, n: usize, mut s: u64) -> Mat<f64> {
    Mat::from_fn(m, n, |_, _| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    })
}

/// Rebuild Q (m×m) = H_0·H_1·…·H_{k-1} from the reflectors stored below the
/// diagonal of `f` (v[0]=1 implicit) and `tau`, by applying each H to the
/// columns of an identity.
fn reconstruct_q(f: &Mat<f64>, tau: &[f64], m: usize, k: usize) -> Mat<f64> {
    let mut q = Mat::from_fn(m, m, |i, j| if i == j { 1.0 } else { 0.0 });
    // apply H_{k-1} … H_0 on the left: Q = H_0(H_1(…(I)))
    for j in (0..k).rev() {
        let t = tau[j];
        if t == 0.0 {
            continue;
        }
        // v: v[j]=1, v[i]=f[i,j] for i>j
        for c in 0..m {
            // w = vᵀ · Q[:,c] over rows j..m
            let mut w = q[(j, c)];
            for i in (j + 1)..m {
                w += f[(i, j)] * q[(i, c)];
            }
            w *= t;
            q[(j, c)] -= w;
            for i in (j + 1)..m {
                q[(i, c)] -= w * f[(i, j)];
            }
        }
    }
    q
}

#[test]
fn factorization_backward_error() {
    for &(m, n) in &[
        (1usize, 1usize),
        (2, 2),
        (3, 3),
        (5, 5),
        (8, 8),
        (31, 31),
        (32, 32),
        (33, 33),
        (64, 64),
        (96, 96),
        (129, 129),
        (256, 256),
        (65, 40), // tall
        (40, 65), // wide
    ] {
        let a = fill(m, n, 0x2545F4914F6CDD1D ^ ((m as u64) << 20) ^ (n as u64));
        let k = m.min(n);
        let mut f = a.clone();
        let mut tau = vec![0.0f64; k];
        qr_factor_in_place(f.as_mut(), &mut tau);

        // R: upper triangle of f (m×n)
        let r = Mat::from_fn(m, n, |i, j| if i <= j { f[(i, j)] } else { 0.0 });
        let q = reconstruct_q(&f, &tau, m, k);

        // ‖A − Q·R‖_max
        let qr = &q * &r;
        let mut berr = 0.0f64;
        for j in 0..n {
            for i in 0..m {
                berr = berr.max((a[(i, j)] - qr[(i, j)]).abs());
            }
        }
        let tol = 1e-12 * (m.max(n).max(4) as f64);
        assert!(berr < tol, "m={m} n={n}: ||A-QR|| = {berr:.2e} >= {tol:.2e}");

        // orthogonality: ‖QᵀQ − I‖_max
        let mut oerr = 0.0f64;
        for c in 0..m {
            for rr in 0..m {
                let mut d = 0.0;
                for i in 0..m {
                    d += q[(i, rr)] * q[(i, c)];
                }
                oerr = oerr.max((d - if rr == c { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(oerr < 1e-11 * (m as f64), "m={m} n={n}: ||QᵀQ-I|| = {oerr:.2e}");

        // |R| agrees with faer's QR (R is unique up to row signs)
        let fq = a.qr();
        let fr = fq.R();
        let mut derr = 0.0f64;
        let mut scale = 0.0f64;
        for j in 0..n {
            for i in 0..=j.min(m - 1) {
                derr = derr.max((f[(i, j)].abs() - fr[(i, j)].abs()).abs());
                scale = scale.max(fr[(i, j)].abs());
            }
        }
        assert!(
            derr < 1e-9 * scale.max(1.0),
            "m={m} n={n}: |R| disagrees with faer by {derr:.2e} (scale {scale:.2e})"
        );
    }
}
