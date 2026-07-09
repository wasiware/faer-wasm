// Complexity verification: measure each op across a size sweep, fit the
// empirical exponent p in time ≈ c · nᵖ, and detect cliff-class jumps.
//   node complexity.mjs <wasm-path>            full sweep, table + fit
//   node complexity.mjs <wasm-path> --gate     smaller sweep, CI mode
//
// Two independent detectors, because they catch different failures:
//
//   1. FITTED EXPONENT — least-squares slope of log t vs log n over the
//      asymptotic sizes (n ≥ 96; smaller sizes are dominated by call
//      overhead and cache effects and systematically understate p). All
//      ops here are Θ(n³); the accepted window [1.5, 3.6] is tight enough
//      that n⁴ behavior fails while tolerating the LU family, which sits
//      near p ≈ 1.8–2.0 at n ≤ 256 (its cubic term has a small constant —
//      n³/3 vs matmul's 2n³ — so quadratic overheads still dominate).
//
//   2. STEP JUMPS — consecutive-size ratio capped at 4 × (n₂/n₁)³. A
//      blocking-threshold misfire is invisible to the exponent fit (the
//      curve is n³ on both sides of a 10–25× constant jump) but trips
//      this immediately. This is exactly how the Schur/EVD cliff at
//      faer's blocking_threshold = 75 was found on 2026-07-09.
//
// gen_evd is exempt from the jump check with an annotation: faer's own
// .eigenvalues() crosses its internal threshold at n = 75 with no public
// way to pass params (guidance: use faer-schur, which ships wasm-tuned
// defaults — see docs/wasm.md §7).
import { readFileSync } from 'node:fs';

const [wasmPath, flag] = process.argv.slice(2);
if (!wasmPath) {
	console.error('usage: node complexity.mjs <wasm-path> [--gate]');
	process.exit(2);
}
const gateMode = flag === '--gate';
const bytes = readFileSync(wasmPath);

const FULL_SIZES = [64, 96, 128, 192, 256];
const GATE_SIZES = [96, 128, 192];
const FIT_MIN_N = 96;
const OPS = [
	['matmul', 'run_matmul'],
	['lu_solve', 'run_lu_solve'],
	['qr', 'run_qr'],
	['svd', 'run_svd'],
	['sa_evd', 'run_sa_evd'],
	['gen_evd', 'run_gen_evd'],
	['schur', 'run_schur'],
	['matmul_c64', 'run_matmul_c64'],
	['lu_solve_c64', 'run_lu_solve_c64'],
	['qr_c64', 'run_qr_c64'],
	['schur_c64', 'run_schur_c64'],
];
const EXP_WINDOW = [1.5, 3.6];
const JUMP_CAP = 4.0;
// exempt from BOTH checks (the cliff also distorts the fitted exponent)
const EXEMPT = {
	gen_evd: 'known cliff: faer .eigenvalues() crosses blocking_threshold=75 internally, no public params — use faer-schur (docs/wasm.md §7)',
};

async function time(exportName, n) {
	let best = Infinity;
	for (let rep = 0; rep < 3; rep++) {
		const { instance } = await WebAssembly.instantiate(bytes, {});
		const e = instance.exports;
		e.setup(n);
		const f = e[exportName];
		let sink = f();
		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		// schur allocates ~6 n² doubles per call through the leak-only bump
		// allocator — cap iterations harder than the generic ops
		const leakCap = Math.floor(150e6 / (8 * 8 * n * n));
		const iters = Math.min(Math.max(Math.ceil(0.1 / per), 3), Math.min(200, Math.max(leakCap, 3)));
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		const ns = ((performance.now() - t0) * 1e6) / iters;
		if (!Number.isFinite(sink)) {
			console.error(`${exportName}(n=${n}): non-finite result`);
			process.exit(1);
		}
		best = Math.min(best, ns);
	}
	return best;
}

function fitSlope(points) {
	const xs = points.map(([n]) => Math.log(n));
	const ys = points.map(([, t]) => Math.log(t));
	const mx = xs.reduce((a, b) => a + b) / xs.length;
	const my = ys.reduce((a, b) => a + b) / ys.length;
	let num = 0, den = 0;
	for (let i = 0; i < xs.length; i++) {
		num += (xs[i] - mx) * (ys[i] - my);
		den += (xs[i] - mx) ** 2;
	}
	return num / den;
}

const sizes = gateMode ? GATE_SIZES : FULL_SIZES;
let failed = false;
console.log(`op            ${sizes.map(n => String(n).padStart(9)).join('')}   fitted p (n≥${FIT_MIN_N})`);
for (const [name, exportName] of OPS) {
	const points = [];
	for (const n of sizes) {
		points.push([n, await time(exportName, n)]);
	}
	const asym = points.filter(([n]) => n >= FIT_MIN_N);
	const p = fitSlope(asym);
	const expOk = p >= EXP_WINDOW[0] && p <= EXP_WINDOW[1];

	let jumpNote = '';
	for (let i = 1; i < points.length; i++) {
		const [n1, t1] = points[i - 1];
		const [n2, t2] = points[i];
		const cap = JUMP_CAP * (n2 / n1) ** 3;
		if (t2 / t1 > cap) {
			if (EXEMPT[name]) {
				jumpNote = `  (jump ${n1}→${n2} = ${(t2 / t1).toFixed(1)}×, exempt: ${EXEMPT[name]})`;
			} else {
				jumpNote = `  CLIFF ${n1}→${n2} = ${(t2 / t1).toFixed(1)}× (cap ${cap.toFixed(1)}×)`;
				failed = true;
			}
		}
	}
	failed ||= !expOk && !EXEMPT[name];
	const times = points.map(([, t]) => (t / 1e6).toFixed(2).padStart(9)).join('');
	console.log(`${name.padEnd(14)}${times}   ${p.toFixed(2)}${expOk || EXEMPT[name] ? '' : ` OUT OF [${EXP_WINDOW}]`}${jumpNote}`);
}
console.log('(times in ms, min-of-3; p = least-squares slope of log t vs log n on the asymptotic sizes)');
process.exit(failed ? 1 : 0);
