// On-runner tuning sweep for the recursive LU's two knobs (crossover,
// trsm_base). Dev-box numbers are noisy and the wrong machine; this runs on
// the same GitHub runner the pyodide head-to-head uses, so the picked
// defaults are chosen where the comparison actually happens.
//   node lu-tune.mjs <bench-wasm>
//
// Trust hardening (2026-07-09): a single sweep on one noisy runner instance
// can crown a fluke winner. So each config is timed over ROUNDS independent
// sweep passes (min across rounds), and the report prints the TOP-3 configs
// per size — if they're tightly bunched the "winner" is a coin-flip and the
// pick should be the simplest, not the nominal fastest.
import { readFileSync, writeFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: node lu-tune.mjs <bench-wasm>');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);
const ROUNDS = 3;

async function time(exportName, n, args) {
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
// widened vs the first sweep: co up to 512 tests whether ANY recursion helps
// at n=512 (co=512 → pure flat, no gemm; co=256 → one split), and tb up to
// 256 tests a non-recursing trsm.
const CROSSOVERS = [64, 96, 128, 192, 256, 384, 512];
const TRSM_BASES = [64, 96, 128, 192, 256];

// aggregate min across ROUNDS for one (export,args) config
async function timeAgg(exportName, n, args) {
	let m = Infinity;
	for (let r = 0; r < ROUNDS; r++) m = Math.min(m, await time(exportName, n, args));
	return m;
}

const out = [];
console.log(`# recursive-LU tuning sweep (ms, min across ${ROUNDS} rounds × min-of-N, on-runner)`);
for (const n of SIZES) {
	const wk = await timeAgg('run_lu_factor_wk', n, [0]);
	const grid = [];
	const seen = new Set();
	for (const co of CROSSOVERS) {
		for (const tb of TRSM_BASES) {
			const eff = Math.min(co, n);
			// when eff >= n there is no recursion, so trsm_base is inert — time
			// it once (tb=64) and skip the duplicates
			const key = eff >= n ? `${eff}:flat` : `${eff}:${tb}`;
			if (seen.has(key)) continue;
			seen.add(key);
			const ms = await timeAgg('run_lu_factor_rec_tuned', n, [eff, tb]);
			grid.push({ crossover: eff, trsm_base: eff >= n ? 0 : tb, ms });
		}
	}
	grid.sort((a, b) => a.ms - b.ms);
	const top3 = grid.slice(0, 3);
	const current = grid.find(g => g.crossover === Math.min(256, n) && (g.trsm_base === 128 || g.trsm_base === 0));
	out.push({ n, wk_ms: wk, current_default_ms: current?.ms ?? null, best: top3[0], top3, grid });
	const spread = ((top3[2]?.ms ?? top3[0].ms) / top3[0].ms - 1) * 100;
	console.log(
		`n=${n}: wk=${wk.toFixed(3)}  current(co256,tb128)=${(current?.ms ?? NaN).toFixed(3)}  ` +
		`TOP3: ` + top3.map(g => `[co=${g.crossover},tb=${g.trsm_base}→${g.ms.toFixed(3)}]`).join(' ') +
		`  (top3 spread ${spread.toFixed(1)}%)`,
	);
}
writeFileSync('lu-tune-results.json', JSON.stringify(out, null, '\t'));
console.log('\nwrote lu-tune-results.json');
