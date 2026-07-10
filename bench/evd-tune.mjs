// EVD (eigvals) phase profile + parameter sweep. Locates faer's 0.3-0.4x
// eigvals deficit: Hessenberg vs QR-iteration split, scalar-lahqr vs blocked
// multishift/AED (the path measured 2-13x slower on wasm, 2026-07-09), and
// whether LAPACK iparmq-style parameters (nibble=14, active-block shift
// counts) repair the multishift path. Counters (AED calls / sweeps) split
// "converges slower" from "each sweep slower".
//   node evd-tune.mjs <bench-wasm>
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node evd-tune.mjs <bench-wasm>');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);

async function time(exportName, n, args = []) {
	let best = Infinity;
	const reps = n >= 512 ? 3 : 4;
	for (let rep = 0; rep < reps; rep++) {
		const { instance } = await WebAssembly.instantiate(bytes, {});
		const e = instance.exports;
		e.setup(n);
		const f = () => e[exportName](...args);
		let sink = f();
		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		const iters = Math.min(Math.max(Math.ceil(0.15 / per), 3), 100);
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		best = Math.min(best, ((performance.now() - t0) * 1e6) / iters); // ns
		if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	}
	return best / 1e6; // ms
}

async function counters(n, args) {
	const { instance } = await WebAssembly.instantiate(bytes, {});
	const e = instance.exports;
	e.setup(n);
	const packed = e.run_eigvals_counters(...args);
	return { aed: Math.floor(packed / 100000), sweeps: packed % 100000 };
}

// [label, blocking, nibble, profile]
const VARIANTS = [
	['default (multishift/AED)', 0, 0, 0],
	['lahqr-pinned (blocking=max)', 1 << 30, 0, 0],
	['nibble=14 (LAPACK)', 0, 14, 0],
	['iparmq shifts/window', 0, 0, 1],
	['nibble=14 + iparmq', 0, 14, 1],
];

const SIZES = [64, 128, 256, 512];
console.log('# EVD eigvals phase profile + parameter sweep (on-runner)');
console.log('# eigenvalues-only pipeline (no Z, want_t=false), min-of-N ms');
for (const n of SIZES) {
	console.log(`\nn=${n}:`);
	const hess = await time('run_hess_only', n);
	const evd = await time('run_gen_evd', n);
	console.log(`  hessenberg only         ${hess.toFixed(3)} ms`);
	console.log(`  faer eigenvalues()      ${evd.toFixed(3)} ms   -> hess is ${((hess / evd) * 100).toFixed(0)}% of eigvals`);
	let baseline = null;
	for (const [label, blocking, nibble, profile] of VARIANTS) {
		const ms = await time('run_eigvals_tuned', n, [blocking, nibble, profile]);
		if (baseline === null) baseline = ms;
		const { aed, sweeps } = await counters(n, [blocking, nibble, profile]);
		const rel = ` (${(baseline / ms).toFixed(2)}x vs default)`;
		console.log(
			`  ${label.padEnd(28)} ${ms.toFixed(3)} ms${rel}  [aed=${aed} sweeps=${sweeps}]`,
		);
	}
}
