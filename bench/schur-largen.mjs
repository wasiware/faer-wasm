// Wasm side of the Schur wasm-vs-native comparison (Schur campaign
// 2026-07-11): the kernel full-Schur pipeline (run_schur_k), the faer-schur
// baseline (run_schur), and the want_t/Z mode split, at n = 64..1024.
// Native side: `cargo run --release --bin native` (schur_k / schur_faer
// rows). Emits the same JSON lines as bench.mjs so the outputs diff.
//   node schur-largen.mjs <bench-wasm> [label]
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
const label = process.argv[3] ?? 'wasm-o3';
if (!wasmPath) {
	console.error('usage: node schur-largen.mjs <bench-wasm> [label]');
	process.exit(2);
}
const bytes = readFileSync(wasmPath);

async function time(exportName, n, args = []) {
	let best = Infinity;
	const reps = n >= 512 ? 2 : 3;
	for (let rep = 0; rep < reps; rep++) {
		const { instance } = await WebAssembly.instantiate(bytes, {});
		const e = instance.exports;
		e.setup(n);
		const f = () => e[exportName](...args);
		let sink = f();
		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		const leakCap = Math.floor(400e6 / (8 * 8 * n * n));
		const iters = Math.min(Math.max(Math.ceil(0.2 / per), 3), Math.min(50, Math.max(leakCap, 3)));
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		best = Math.min(best, ((performance.now() - t0) * 1e6) / iters);
		if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	}
	return best;
}

const SIZES = [64, 128, 256, 512, 1024];
for (const n of SIZES) {
	const k = await time('run_schur_k', n);
	console.log(JSON.stringify({ target: label, op: 'schur_k', n, ns: Math.round(k) }));
	const f = await time('run_schur', n);
	console.log(JSON.stringify({ target: label, op: 'schur_faer', n, ns: Math.round(f) }));
}

console.log('\n# want_t/Z mode split (min-of-2, ms): 0=eigvals 1=+T 2=+Z 3=T+Z');
for (const n of SIZES) {
	const ms = [];
	for (const mode of [0, 1, 2, 3]) {
		ms.push((await time('run_schur_k_mode', n, [mode])) / 1e6);
	}
	const tShare = (((ms[1] - ms[0]) / ms[3]) * 100).toFixed(0);
	const zShare = (((ms[2] - ms[0]) / ms[3]) * 100).toFixed(0);
	console.log(
		`n=${n}: ${ms.map((m) => m.toFixed(2)).join(' / ')}  (T share ${tShare}%, Z share ${zShare}%)`,
	);
}
