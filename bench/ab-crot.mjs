// Same-machine A/B of two bench wasm builds (Schur close-out, 2026-07-12).
//
// Motivation: GitHub Actions runners are heterogeneous — between two runs
// with NO change to the f64 kernels, schur_k@256 drifted 81.9→93.6 ms
// (runs 29157035070 vs 29172715791). Cross-run ratios therefore cannot
// judge a code change; this script times both builds interleaved on one
// machine, the same way the replication gate alternates against scipy.
//
//   node ab-crot.mjs <a.wasm> <b.wasm> [labelA] [labelB]
//
// Rows: run_schur_c64_k (the op under test) and run_schur_k (untouched
// between the builds — a control that should read ~1.00x; if it doesn't,
// the machine is too noisy to conclude anything).
import { readFileSync } from 'node:fs';

const [pa, pb, la = 'A', lb = 'B'] = process.argv.slice(2);
if (!pa || !pb) {
	console.error('usage: node ab-crot.mjs <a.wasm> <b.wasm> [labelA] [labelB]');
	process.exit(2);
}
const bytesA = readFileSync(pa);
const bytesB = readFileSync(pb);

async function timeOnce(bytes, exportName, n) {
	const { instance } = await WebAssembly.instantiate(bytes, {});
	const e = instance.exports;
	e.setup(n);
	let sink = e[exportName](); // warm + compile
	const iters = n >= 256 ? 2 : 8;
	const t0 = performance.now();
	for (let i = 0; i < iters; i++) sink += e[exportName]();
	if (!Number.isFinite(sink)) throw new Error(`${exportName}(n=${n}): non-finite`);
	return (performance.now() - t0) / iters;
}

const fmt = (ts) => {
	ts.sort((x, y) => x - y);
	return { med: ts[2], lo: ts[0], hi: ts[4] };
};

for (const op of ['run_schur_c64_k', 'run_schur_k']) {
	const tag = op === 'run_schur_k' ? ' (control — code identical in both builds)' : '';
	console.log(`\n## ${op}${tag} — 5 alternating rounds, median [min..max] ms`);
	for (const n of [64, 128, 256]) {
		const ta = [];
		const tb = [];
		for (let r = 0; r < 5; r++) {
			ta.push(await timeOnce(bytesA, op, n));
			tb.push(await timeOnce(bytesB, op, n));
		}
		const a = fmt(ta);
		const b = fmt(tb);
		const sep = a.hi < b.lo || b.hi < a.lo;
		const ratio = (b.med / a.med).toFixed(2);
		console.log(
			`n=${n}: ${la} ${a.med.toFixed(2)} [${a.lo.toFixed(2)}..${a.hi.toFixed(2)}]  ` +
				`${lb} ${b.med.toFixed(2)} [${b.lo.toFixed(2)}..${b.hi.toFixed(2)}]  ` +
				`-> ${sep ? (a.med < b.med ? `${la} WINS` : `${lb} WINS`) : 'OVERLAP'} (${lb}/${la} = ${ratio}x)`,
		);
	}
}
