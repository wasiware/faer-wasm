// SVD recursion_threshold sweep on the runner. Deep research (2026-07-10)
// fingered faer's default divide-and-conquer recursion_threshold=128 as the
// root cause of SVD losing to scipy at small n: bidiagonal blocks up to 127
// go through the SCALAR qr_algorithm (Givens vector accumulation, no SIMD),
// where LAPACK dbdsdc uses ~25-element leaves + gemm merges. This sweeps the
// knob (low AND high) against faer's own default to test whether the fix
// survives on the reference machine — the dev box has misled on this exact
// knob, so the runner is the arbiter.
//   node svd-tune.mjs <bench-wasm>
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node svd-tune.mjs <bench-wasm>');
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

// thresholds to probe per size: 0 == faer default (128), then a low sweep
// (test "smaller leaves + more gemm merges") and a high sweep (test "one big
// scalar leaf, skip the DC merges entirely").
const SWEEP = {
	64: [0, 8, 16, 32, 48, 96, 128],
	128: [0, 8, 16, 32, 48, 96, 192, 256],
	256: [0, 32, 64, 96, 192, 384, 512],
	512: [0, 96, 192, 256, 384, 640, 1024],
};

console.log('# SVD recursion_threshold sweep (on-runner). rt=0 is faer default (128).');
console.log('# ms = min-of-N wall time for full SVD with both singular-vector sets.');
for (const n of Object.keys(SWEEP).map(Number)) {
	console.log(`\nn=${n}:`);
	let baseline = null;
	for (const rt of SWEEP[n]) {
		const ms = await time('run_svd_tuned', n, [rt]);
		if (rt === 0) baseline = ms;
		const rel = baseline ? ` (${(baseline / ms).toFixed(2)}× vs default)` : '';
		const tag = rt === 0 ? 'default(128)' : `rt=${rt}`;
		console.log(`  ${tag.padEnd(12)} ${ms.toFixed(3)} ms${rel}`);
	}
}
