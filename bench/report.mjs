// Merge benchmark JSON-lines files into the markdown tables used in
// docs/benchmarks-*.md:
//   node report.mjs native-o3.jsonl wasm-z-plain.jsonl ... > tables.md
// The first file is the baseline; every other target is reported as its
// time plus the ratio vs the baseline at the same (op, n).
import { readFileSync } from 'node:fs';

const files = process.argv.slice(2);
if (files.length < 2) {
	console.error('usage: node report.mjs <baseline.jsonl> <other.jsonl>...');
	process.exit(2);
}

const rows = files.flatMap(f =>
	readFileSync(f, 'utf8').trim().split('\n').map(l => JSON.parse(l)),
);
const targets = [...new Set(rows.map(r => r.target))];
const baseline = targets[0];
const ops = [...new Set(rows.map(r => r.op))];
const get = (t, op, n) => rows.find(r => r.target === t && r.op === op && r.n === n)?.ns;

const fmt = ns =>
	ns >= 1e6 ? `${(ns / 1e6).toFixed(2)} ms` : `${(ns / 1e3).toFixed(1)} µs`;

const ratios = {};
for (const op of ops) {
	const ns = [...new Set(rows.filter(r => r.op === op).map(r => r.n))].sort((a, b) => a - b);
	console.log(`### ${op}\n`);
	console.log(`| n | ${baseline} | ${targets.slice(1).map(t => `${t} (×)`).join(' | ')} |`);
	console.log(`| -: | -: | ${targets.slice(1).map(() => '-:').join(' | ')} |`);
	for (const n of ns) {
		const base = get(baseline, op, n);
		const cells = targets.slice(1).map(t => {
			const v = get(t, op, n);
			if (v == null || base == null) return '—';
			const ratio = v / base;
			(ratios[t] ??= []).push(ratio);
			return `${fmt(v)} (${ratio.toFixed(2)}×)`;
		});
		console.log(`| ${n} | ${fmt(base)} | ${cells.join(' | ')} |`);
	}
	console.log('');
}

console.log(`### Geometric-mean slowdown vs ${baseline}\n`);
console.log('| target | geomean × |');
console.log('| - | -: |');
for (const t of targets.slice(1)) {
	const g = Math.exp(ratios[t].reduce((s, r) => s + Math.log(r), 0) / ratios[t].length);
	console.log(`| ${t} | ${g.toFixed(2)}× |`);
}
