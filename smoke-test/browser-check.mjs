// Browser gate: run the wasm probes in a REAL browser (headless Chrome) and
// compare against the same references as the node gate.
//   node browser-check.mjs <wasm-path> <variant>
//
// Dependency-free on purpose (this repo has no npm deps): we launch Chrome
// with --remote-debugging-port=0 and speak raw Chrome DevTools Protocol over
// node 22's built-in WebSocket. The wasm bytes travel into the page as
// base64 inside a Runtime.evaluate expression; the probe results come back
// by value. Chrome is found via $CHROME or common executable names.
//
// This is what upgrades "runs in real browsers" from *stated* to
// CI-enforced: node and Chrome both embed V8, but the wasm engine
// configuration differs (notably relaxed-SIMD availability), so the
// full-relaxed variant passing here is a genuinely new fact per push.
import { readFileSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { spawn, execSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { reference, required as requiredByVariant } from './references.mjs';

const [wasmPath, variant] = process.argv.slice(2);
const required = requiredByVariant[variant];
if (!wasmPath || !required) {
	console.error('usage: node browser-check.mjs <wasm-path> <variant>');
	process.exit(2);
}

function findChrome() {
	if (process.env.CHROME) return process.env.CHROME;
	for (const c of ['/opt/pw-browsers/chromium', 'google-chrome', 'chromium-browser', 'chromium']) {
		if (c.startsWith('/') ? existsSync(c) : canRun(c)) return c;
	}
	console.error('no Chrome/Chromium found; set $CHROME');
	process.exit(2);
}
function canRun(cmd) {
	try {
		execSync(`command -v ${cmd}`, { stdio: 'ignore' });
		return true;
	} catch {
		return false;
	}
}

const chrome = findChrome();
const profile = mkdtempSync(join(tmpdir(), 'wasm-gate-chrome-'));
const proc = spawn(chrome, [
	'--headless=new',
	'--no-sandbox',
	'--disable-gpu',
	'--disable-dev-shm-usage',
	`--user-data-dir=${profile}`,
	'--remote-debugging-port=0',
	'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

const cleanup = () => {
	try { proc.kill('SIGKILL'); } catch {}
	try { rmSync(profile, { recursive: true, force: true }); } catch {}
};
process.on('exit', cleanup);

// 1. wait for the DevTools websocket URL on stderr
const wsUrl = await new Promise((resolve, reject) => {
	let buf = '';
	const timer = setTimeout(() => reject(new Error(`Chrome did not start:\n${buf}`)), 20000);
	proc.stderr.on('data', (d) => {
		buf += d;
		const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
		if (m) {
			clearTimeout(timer);
			resolve(m[1]);
		}
	});
	proc.on('exit', (code) => reject(new Error(`Chrome exited early (${code}):\n${buf}`)));
});

// 2. raw CDP client
const ws = new WebSocket(wsUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let nextId = 1;
const pending = new Map();
ws.onmessage = (ev) => {
	const msg = JSON.parse(ev.data);
	if (msg.id && pending.has(msg.id)) {
		const { res, rej } = pending.get(msg.id);
		pending.delete(msg.id);
		msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
	}
};
function cdp(method, params = {}, sessionId) {
	const id = nextId++;
	ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
	return new Promise((res, rej) => pending.set(id, { res, rej }));
}

// 3. open a page and run the probes in it
const { targetId } = await cdp('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await cdp('Target.attachToTarget', { targetId, flatten: true });

const b64 = readFileSync(wasmPath).toString('base64');
const expression = `(async () => {
	const bin = atob(${JSON.stringify(b64)});
	const bytes = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
	const t0 = performance.now();
	const { instance } = await WebAssembly.instantiate(bytes.buffer, {});
	const out = { _instantiate_ms: Math.round(performance.now() - t0) };
	for (const name of ${JSON.stringify(required)}) {
		if (typeof instance.exports[name] !== 'function') { out[name] = '__missing__'; continue; }
		const t = performance.now();
		out[name] = instance.exports[name]();
		out['_' + name + '_ms'] = Math.round(performance.now() - t);
	}
	return out;
})()`;

const evalRes = await cdp('Runtime.evaluate', {
	expression,
	awaitPromise: true,
	returnByValue: true,
	timeout: 120000,
}, sessionId);

if (evalRes.exceptionDetails) {
	// a CompileError here on full-relaxed means this browser lacks relaxed-SIMD
	console.error(`[browser:${variant}] evaluation threw:`, JSON.stringify(evalRes.exceptionDetails.exception?.description ?? evalRes.exceptionDetails, null, 2));
	process.exit(1);
}

const out = evalRes.result.value;
const version = (await cdp('Browser.getVersion')).product;
console.log(`[browser:${variant}] ${version}, instantiate ${out._instantiate_ms} ms`);
let failed = false;
for (const name of required) {
	const got = out[name];
	const want = reference[name];
	const ok = Object.is(got, want);
	console.log(`[browser:${variant}] ${name} = ${got} (want ${want}) ${ok ? 'ok' : 'FAIL'}  [${out['_' + name + '_ms']} ms]`);
	failed ||= !ok;
}
ws.close();
cleanup();
process.exit(failed ? 1 : 0);
