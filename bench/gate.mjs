// Efficiency gate: machine-robust performance checks for the wasm build.
//   node gate.mjs <wasm-path>            check against expected-ratios.json
//   node gate.mjs <wasm-path> --record   print a fresh expected-ratios.json
//
// Absolute times are useless on shared CI runners, so everything here is a
// ratio measured within one process on one machine:
//
//   1. op/matmul at n=128 vs the recorded expectation, within a ×3 band
//      either way — catches cliff-class regressions (the blocking misfires
//      we measured in 2026-07 were 3–10×) while ignoring runner noise.
//   2. op(256)/op(128) inside a wide O(n³) window [3, 26] — catches
//      complexity-class blowups without any recorded baseline.
//   3. tuned-vs-default factor kernels at n=128: the docs/wasm.md §7
//      guidance (unblocked LU, panel-1 QR) must not become *slower* than
//      faer's defaults — if a re-pin flips this, the recipe is stale.
//
// All measurements are min-of-3 with ~120ms per rep, fresh instance per op
// (the bump allocator leaks; re-instantiation resets the heap).
import { readFileSync } from 'node:fs';

const [wasmPath, flag] = process.argv.slice(2);
if (!wasmPath) {
	console.error('usage: node gate.mjs <wasm-path> [--record]');
	process.exit(2);
}
const record = flag === '--record';
const bytes = readFileSync(wasmPath);

const RATIO_OPS = ['lu_solve', 'qr', 'svd', 'sa_evd', 'gen_evd', 'schur', 'matmul_c64', 'lu_solve_c64', 'qr_c64', 'schur_c64'];
const SCALING_OPS = ['matmul', 'lu_solve', 'qr', 'matmul_c64'];
const BAND = 3.0;               // op/matmul drift allowed vs recorded
const SCALE_WINDOW = [3, 26];   // op(256)/op(128) window (O(n³) ≈ 8)
const TUNED_SLACK = 1.5;        // tuned kernels may be at most 1.5× default

async function fresh() {
	const { instance } = await WebAssembly.instantiate(bytes, {});
	return instance.exports;
}

async function time(exportName, n, args = []) {
	let best = Infinity;
	for (let rep = 0; rep < 3; rep++) {
		const e = await fresh();
		e.setup(n);
		const f = () => e[exportName](...args);
		let sink = f(); // warmup + tier-up
		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		const leakCap = Math.floor(250e6 / (4 * 8 * n * n));
		const iters = Math.min(Math.max(Math.ceil(0.12 / per), 3), Math.min(300, Math.max(leakCap, 3)));
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

let failed = false;
const check = (label, ok, detail) => {
	console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}  ${detail}`);
	failed ||= !ok;
};

// --- 1. op/matmul ratios at n=128
const mm128 = await time('run_matmul', 128);
const ratios = {};
for (const op of RATIO_OPS) {
	ratios[op] = (await time(`run_${op}`, 128)) / mm128;
}

if (record) {
	console.log(JSON.stringify({
		_comment: 'op/matmul wall-time ratios at n=128, node/V8, opt-level 3 wasm build. Recorded 2026-07-09; gate.mjs allows x3 drift either way. Re-record after intentional perf changes.',
		...Object.fromEntries(Object.entries(ratios).map(([k, v]) => [k, Math.round(v * 100) / 100])),
	}, null, '\t'));
	process.exit(0);
}

const expected = JSON.parse(readFileSync(new URL('./expected-ratios.json', import.meta.url)));
for (const op of RATIO_OPS) {
	const got = ratios[op];
	const want = expected[op];
	const ok = got < want * BAND && got > want / BAND;
	check(`ratio ${op}/matmul @128`, ok, `${got.toFixed(2)} (recorded ${want}, band ×${BAND})`);
}

// --- 2. O(n³) scaling windows
for (const op of SCALING_OPS) {
	const t128 = op === 'matmul' ? mm128 : await time(`run_${op}`, 128);
	const t256 = await time(`run_${op}`, 256);
	const r = t256 / t128;
	const ok = r >= SCALE_WINDOW[0] && r <= SCALE_WINDOW[1];
	check(`scaling ${op} 256/128`, ok, `${r.toFixed(1)} (window ${SCALE_WINDOW[0]}–${SCALE_WINDOW[1]})`);
}

// --- 3. docs/wasm.md §7 guidance stays valid: tuned must not lose to default
const luDefault = await time('run_lu_factor_tuned', 128, [0, 0]);
const luTuned = await time('run_lu_factor_tuned', 128, [128, 0]); // recursion_threshold ≥ n
check('LU unblocked (rt≥n) vs default @128', luTuned <= luDefault * TUNED_SLACK,
	`tuned/default = ${(luTuned / luDefault).toFixed(2)} (must be ≤ ${TUNED_SLACK})`);

const qrDefault = await time('run_qr_factor_tuned', 128, [0, 0]);
const qrTuned = await time('run_qr_factor_tuned', 128, [1, 1 << 30]); // panel width 1
check('QR panel-1 vs default @128', qrTuned <= qrDefault * TUNED_SLACK,
	`tuned/default = ${(qrTuned / qrDefault).toFixed(2)} (must be ≤ ${TUNED_SLACK})`);

process.exit(failed ? 1 : 0);
