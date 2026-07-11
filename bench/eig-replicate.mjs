// Replication gate for the eigvals "wins" (architect challenge 2026-07-11:
// "0.8 -> 1.14 -> 1.04 doesn't make sense as a pattern... I think they are
// noise"). Cross-run variance on the eig rows has measured larger than the
// claimed margins (same op@512: 598/836/854 ms across runner instances), so
// single-run ratios are below our evidence bar. This runs R independent
// rounds in ONE job, ALTERNATING faer and scipy measurements (so slow-machine
// drift hits both sides), and reports per-row median + min..max range. A win
// claim requires the ranges to separate (faer.max < scipy.min).
//   PYODIDE_PATH=<pyodide.mjs> node eig-replicate.mjs <bench-wasm>
import { readFileSync } from 'node:fs';

const wasmPath = process.argv[2];
if (!wasmPath) {
	console.error('usage: PYODIDE_PATH=<pyodide.mjs> node eig-replicate.mjs <bench-wasm>');
	process.exit(2);
}
const ROUNDS = 5;
const SIZES = [64, 128, 256, 512];
const FAER_OPS = [
	['eigvals_hk', 'run_eigvals_hk'],
	['eigvals_wk', 'run_eigvals_wk'],
];
const bytes = readFileSync(wasmPath);

async function timeFaer(exportName, n) {
	// one fresh instance, min over an adaptive iteration count (one "round")
	const { instance } = await WebAssembly.instantiate(bytes, {});
	const e = instance.exports;
	e.setup(n);
	const f = () => e[exportName]();
	let sink = f();
	let t0 = performance.now();
	sink += f();
	const per = Math.max((performance.now() - t0) / 1e3, 1e-9);
	const iters = Math.min(Math.max(Math.ceil(0.15 / per), 3), 100);
	t0 = performance.now();
	for (let i = 0; i < iters; i++) sink += f();
	if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	return ((performance.now() - t0) * 1e6) / iters / 1e6; // ms
}

const { loadPyodide } = await import(process.env.PYODIDE_PATH ?? 'pyodide');
const py = await loadPyodide();
await py.loadPackage(['numpy'], { messageCallback: () => {} });
await py.runPythonAsync(`
import time
import numpy as np

def setup(n):
    global a
    rng = np.random.default_rng(0x9E3779B9 ^ n)
    a = rng.uniform(-1, 1, (n, n))

def bench_once():
    f = lambda: np.linalg.eigvals(a)
    f()
    t0 = time.perf_counter(); f()
    per = max(time.perf_counter() - t0, 1e-9)
    iters = min(max(int(0.15 / per) + 1, 3), 100)
    t0 = time.perf_counter()
    for _ in range(iters):
        f()
    return (time.perf_counter() - t0) / iters * 1e3  # ms
`);

const stats = (xs) => {
	const s = [...xs].sort((a, b) => a - b);
	return { min: s[0], med: s[Math.floor(s.length / 2)], max: s[s.length - 1] };
};
const fmt = (r) => `${r.med.toFixed(2)} [${r.min.toFixed(2)}..${r.max.toFixed(2)}]`;

console.log(`# eig replication gate: ${ROUNDS} alternating rounds, median [min..max] ms`);
for (const n of SIZES) {
	await py.runPythonAsync(`setup(${n})`);
	const faer = Object.fromEntries(FAER_OPS.map(([k]) => [k, []]));
	const scipy = [];
	for (let r = 0; r < ROUNDS; r++) {
		for (const [key, exp] of FAER_OPS) {
			faer[key].push(await timeFaer(exp, n));
		}
		scipy.push(await py.runPythonAsync('bench_once()'));
	}
	console.log(`\nn=${n}: scipy eigvals ${fmt(stats(scipy))}`);
	const sc = stats(scipy);
	for (const [key] of FAER_OPS) {
		const st = stats(faer[key]);
		const verdict =
			st.max < sc.min
				? `WIN (separated, ${(sc.med / st.med).toFixed(2)}x)`
				: st.min > sc.max
					? `LOSS (separated, ${(sc.med / st.med).toFixed(2)}x)`
					: `OVERLAP (median ratio ${(sc.med / st.med).toFixed(2)}x) — no claim`;
		console.log(`  ${key.padEnd(11)} ${fmt(st)}  -> ${verdict}`);
	}
}
