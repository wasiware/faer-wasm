// Shared reference values + per-variant required exports for the wasm gate.
// Consumed by check.mjs (node) and browser-check.mjs (headless Chrome).
export const reference = {
	matmul_trace: 114,
	lu_solve_sum: 0.8857142857142857,   // 31/35
	qr_svd_evd_probe: 1.9483450492039642,
	schur_probe: 11,        // faer-schur real f64 property score: 6 checks + m=5 (see smoke-test/src/lib.rs)
	schur_probe_cplx: 3,    // faer-schur c64 property score; guards patches/pulp/0003 (relaxed-simd complex mul fix)
	dense_f64_probe: 26,    // foundation gate: LU/QR/LLT/SVD/EVD f64 property score, n=33+96 (see src/dense_probes.rs)
	dense_c64_probe: 24,    // foundation gate: same in c64
};

export const required = {
	'matmul': ['matmul_trace'],
	'lu': ['matmul_trace', 'lu_solve_sum'],
	'full': ['matmul_trace', 'lu_solve_sum', 'qr_svd_evd_probe', 'schur_probe', 'schur_probe_cplx', 'dense_f64_probe', 'dense_c64_probe'],
	'full-relaxed': ['matmul_trace', 'lu_solve_sum', 'qr_svd_evd_probe', 'schur_probe', 'schur_probe_cplx', 'dense_f64_probe', 'dense_c64_probe'],
};
