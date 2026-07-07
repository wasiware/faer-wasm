import { readFileSync } from 'node:fs';
const wasm = readFileSync(new URL(process.argv[2] ?? './target/wasm32-unknown-unknown/release/consumer.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(wasm, {});
const e = instance.exports;
console.log('exports:', Object.keys(e).join(', '));
if (e.matmul_trace) console.log('matmul_trace =', e.matmul_trace(), '(expected 78: trace of A*B)');
if (e.lu_solve_sum) console.log('lu_solve_sum =', e.lu_solve_sum());
if (e.qr_svd_evd_probe) console.log('qr_svd_evd_probe =', e.qr_svd_evd_probe());
