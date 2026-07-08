// Wasm side of the wasm-vs-native benchmark.
//   node bench.mjs <wasm-path> <label>            full run, JSON lines on stdout
//   node bench.mjs <wasm-path> <label> --smoke    tiny run (CI): asserts finite results
//
// Mirrors bench/src/bin/native.rs: same ops, sizes, and adaptive-iteration
// logic. The module leaks (bump allocator), so it is re-instantiated per
// size and iterations are capped by projected leak.
import { readFileSync } from 'node:fs';

const [wasmPath, label, flag] = process.argv.slice(2);
if (!wasmPath || !label) {
	console.error('usage: node bench.mjs <wasm-path> <label> [--smoke]');
	process.exit(2);
}
const smoke = flag === '--smoke';
const bytes = readFileSync(wasmPath);

const OPS = [
	['matmul', 'run_matmul', 256],
	['lu_solve', 'run_lu_solve', 256],
	['qr', 'run_qr', 256],
	['svd', 'run_svd', 256],
	['sa_evd', 'run_sa_evd', 256],
	['gen_evd', 'run_gen_evd', 128],
];
const SIZES = smoke ? [16] : [32, 64, 128, 256];

for (const n of SIZES) {
	for (const [name, exportName, maxN] of OPS) {
		if (n > maxN) continue;
		// fresh instance per op: resets the leaked heap, keeps peak memory bounded
		const { instance } = await WebAssembly.instantiate(bytes, {});
		const e = instance.exports;
		e.setup(n);
		const f = e[exportName];

		let sink = f(); // warmup (also triggers engine tier-up)
		if (smoke) {
			if (!Number.isFinite(sink)) {
				console.error(`${name}: non-finite result`);
				process.exit(1);
			}
			console.log(`[smoke] ${name}(n=${n}) ok`);
			continue;
		}

		let t0 = performance.now();
		sink += f();
		const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
		const leakCap = Math.floor(250e6 / (4 * 8 * n * n));
		const iters = Math.min(Math.max(Math.ceil(0.15 / per), 3), Math.min(500, Math.max(leakCap, 3)));
		t0 = performance.now();
		for (let i = 0; i < iters; i++) sink += f();
		const ns = ((performance.now() - t0) * 1e6) / iters;
		if (!Number.isFinite(sink)) {
			console.error(`${name}: non-finite result`);
			process.exit(1);
		}
		console.log(JSON.stringify({ target: label, op: name, n, ns: Math.round(ns * 10) / 10 }));
	}
}
