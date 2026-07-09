// Calibration for the dense-probe tolerances (dense_probes.rs): prints the
// raw error magnitudes per check on native so thresholds can be verified to
// sit >= 2 orders of magnitude above reality. Not part of the gate.
fn main() {
    const F64_NAMES: [&str; 13] = [
        "lu_solve", "qr_recon", "qr_orth", "llt_recon", "llt_solve", "svd_recon", "svd_u_orth",
        "svd_v_orth", "sv_sorted", "evd_resid", "evd_orth", "eig_trace", "sa_eig_trace",
    ];
    const C64_NAMES: [&str; 12] = [
        "lu_solve", "qr_recon", "qr_orth", "llt_recon", "llt_solve", "svd_recon", "svd_u_orth",
        "svd_v_orth", "sv_sorted", "evd_resid", "evd_orth", "eig_trace",
    ];
    for n in [33usize, 96] {
        let e = consumer::dense_probes_f64_errors(n);
        for (name, v) in F64_NAMES.iter().zip(e) {
            println!("f64 n={n:3} {name:<12} {v:9.2e}");
        }
        let e = consumer::dense_probes_c64_errors(n);
        for (name, v) in C64_NAMES.iter().zip(e) {
            println!("c64 n={n:3} {name:<12} {v:9.2e}  (squared)");
        }
    }
}
