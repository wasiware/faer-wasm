// Does the recursive LU EVER beat pure-flat on wasm? The routine sweep
// (lu-tune.mjs) only goes to 512, where pure-flat wins/ties — so recursion
// is unproven there. This probes the large-n regime (n up to 1024) where
// theory says gemm-fed recursion *should* eventually pay off as the flat
// panel's cache behavior degrades. If recursion still doesn't clearly beat
// pure-flat here, it's dead weight and should be removed.
//   node lu-largen.mjs <bench-wasm>
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node lu-largen.mjs <bench-wasm>');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);
const ROUNDS = 3;

async function time(exportName, n, args) {
	let best = Infinity;
	const reps = 3;
	for (let rep = 0; rep < reps; rep++) {
		const { instance } = await WebAssembly.instantiate(bytes, {});
		const e = instance.exports;
		e.setup(n);
		const f = () => e[exportName](...args);
		let sink = f();
		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		const leakCap = Math.floor(400e6 / (8 * 8 * n * n)); // bigger budget for n=1024
		const iters = Math.min(Math.max(Math.ceil(0.2 / per), 3), Math.min(50, Math.max(leakCap, 3)));
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		best = Math.min(best, ((performance.now() - t0) * 1e6) / iters);
		if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	}
	return best / 1e6;
}
async function timeAgg(exportName, n, args) {
	let m = Infinity;
	for (let r = 0; r < ROUNDS; r++) m = Math.min(m, await time(exportName, n, args));
	return m;
}

const SIZES = [512, 640, 768, 1024];
console.log(`# LU large-n: pure-flat vs recursion (ms, min across ${ROUNDS} rounds, on-runner)`);
for (const n of SIZES) {
	// pure flat: crossover >= n so lu_rec hits the base case immediately
	const flat = await timeAgg('run_lu_factor_rec_tuned', n, [n, 256]);
	// recursion at a few base widths (tb=256, the tuned trsm base)
	const rec256 = await timeAgg('run_lu_factor_rec_tuned', n, [256, 256]);
	const rec384 = await timeAgg('run_lu_factor_rec_tuned', n, [384, 256]);
	const rec512 = await timeAgg('run_lu_factor_rec_tuned', n, [512, 256]);
	const wk = await timeAgg('run_lu_factor_wk', n, [0]);
	const bestRec = Math.min(rec256, rec384, rec512);
	const verdict = bestRec < flat * 0.97 ? `RECURSION WINS by ${((flat / bestRec - 1) * 100).toFixed(1)}%`
		: bestRec > flat * 1.03 ? `flat wins by ${((bestRec / flat - 1) * 100).toFixed(1)}%`
		: 'tie (within 3%)';
	console.log(
		`n=${n}: flat=${flat.toFixed(2)}  rec[256/384/512]=${rec256.toFixed(2)}/${rec384.toFixed(2)}/${rec512.toFixed(2)}  ` +
		`wk-blocked=${wk.toFixed(2)}  →  ${verdict}`,
	);
}
