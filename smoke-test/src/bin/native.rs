// Native side of the cross-target determinism gate: print the probe values
// as exact f64 bit patterns. determinism.mjs compares these against the same
// computations run in the wasm build — bit-for-bit, no tolerance.
fn main() {
    println!("matmul_trace={:016x}", consumer::matmul_trace().to_bits());
    println!("lu_solve_sum={:016x}", consumer::lu_solve_sum().to_bits());
    println!(
        "qr_svd_evd_probe={:016x}",
        consumer::qr_svd_evd_probe().to_bits()
    );
    println!("schur_probe={:016x}", consumer::schur_probe().to_bits());
    println!(
        "schur_probe_cplx={:016x}",
        consumer::schur_probe_cplx().to_bits()
    );
    println!(
        "dense_f64_probe={:016x}",
        consumer::dense_f64_probe().to_bits()
    );
    println!(
        "dense_c64_probe={:016x}",
        consumer::dense_c64_probe().to_bits()
    );
}
