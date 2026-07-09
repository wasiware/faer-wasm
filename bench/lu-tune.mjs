// On-runner tuning sweep for the recursive LU's two knobs (crossover,
// trsm_base). Dev-box numbers are noisy and the wrong machine; this runs on
// the same GitHub runner the pyodide head-to-head uses, so the picked
// defaults are chosen where the comparison actually happens.
//   node lu-tune.mjs <bench-wasm>
// Emits a JSON blob + a markdown table; the winning (crossover, trsm_base)
// per size is what should be baked into kernels/src/lu.rs.
import { readFileSync, writeFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node lu-tune.mjs <bench-wasm>');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);

async function time(exportName, n, args) {
	let best = Infinity;
	const reps = n >= 512 ? 3 : 4; // a touch more than the head-to-head — tuning wants tighter noise
	for (let rep = 0; rep < reps; rep++) {
		const { instance } = await WebAssembly.instantiate(bytes, {});
		const e = instance.exports;
		e.setup(n);
		const f = () => e[exportName](...args);
		let sink = f();
		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		const leakCap = Math.floor(150e6 / (8 * 8 * n * n));
		const iters = Math.min(Math.max(Math.ceil(0.2 / per), 5), Math.min(200, Math.max(leakCap, 5)));
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		best = Math.min(best, ((performance.now() - t0) * 1e6) / iters);
		if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	}
	return best / 1e6; // ms
}

const SIZES = [192, 256, 384, 512];
const CROSSOVERS = [64, 96, 128, 160, 192, 256];
const TRSM_BASES = [32, 48, 64, 96, 128];

const out = [];
console.log('# recursive-LU tuning sweep (ms, min-of-N, on-runner)');
for (const n of SIZES) {
	// baselines for context
	const wk = await time('run_lu_factor_wk', n, [0]);
	let best = { ms: Infinity, crossover: null, trsm_base: null };
	const grid = [];
	for (const co of CROSSOVERS) {
		for (const tb of TRSM_BASES) {
			// crossover >= n means pure base case; skip the redundant duplicates
			const eff = Math.min(co, n);
			const ms = await time('run_lu_factor_rec_tuned', n, [eff, tb]);
			grid.push({ crossover: eff, trsm_base: tb, ms });
			if (ms < best.ms) best = { ms, crossover: eff, trsm_base: tb };
		}
	}
	const rec = grid.filter(g => g.crossover === Math.min(128, n) && g.trsm_base === 64)[0];
	out.push({ n, wk_ms: wk, current_default_ms: rec?.ms ?? null, best, grid });
	console.log(
		`n=${n}: wk=${wk.toFixed(3)}  current(co128,tb64)=${(rec?.ms ?? NaN).toFixed(3)}  ` +
		`BEST co=${best.crossover} tb=${best.trsm_base} → ${best.ms.toFixed(3)} ms ` +
		`(${(((rec?.ms ?? best.ms) / best.ms - 1) * 100).toFixed(1)}% over current)`,
	);
}
writeFileSync('lu-tune-results.json', JSON.stringify(out, null, '\t'));
console.log('\nwrote lu-tune-results.json');
