# GitHub issue to file — pure copy-paste, no fork, no Codeberg

Where: https://github.com/sarah-quinones/faer-rs/issues/new

Paste the title and body below verbatim. The two patches are inlined, so
the issue is self-contained — the maintainer can `git am` them directly.

---

## Title

```
faer fails to compile on all 32-bit targets (wasm32, armv7, i686): `(n >> 32)` on usize — 4-line fix, tests, and wasm CI attached
```

## Body

faer 0.24.4 does not compile for **any 32-bit target** — `wasm32-unknown-unknown`, `armv7`, `i686` — with any feature set:

```
error: this arithmetic operation will overflow
   --> faer/src/./operator/eigen/mod.rs:319:13
    |
319 |             let n1 = (n >> 32) as u32;
    |                      ^^^^^^^^^ attempt to shift right by `32_i32`, which would overflow
```

(also at `operator/eigen/mod.rs:771`, `operator/self_adjoint_eigen/mod.rs:108`, `operator/svd/mod.rs:159`)

The zero-`v0` fallback in the matrix-free partial eigen/svd operators splits `n: usize` into two `u32` halves with `(n >> 32)`, which is a compile-time overflow error when `usize` is 32 bits. The fix is to widen before shifting — a no-op on 64-bit targets:

```diff
-			let n1 = (n >> 32) as u32;
+			let n1 = ((n as u64) >> 32) as u32;
```

**Verified:** with this applied, the full dense suite (matmul, partial-pivot LU solve, QR, SVD, self-adjoint EVD, general complex EVD) builds for `wasm32-unknown-unknown` with `default-features = false, features = ["linalg"]` and runs under node **bit-identical to native x86-64** (rustc 1.94.1) — 51 KiB (matmul-only) up to ~396 KiB (full suite) at `opt-level = "z"` + fat LTO. Related: #222 shows faer already has wasm users in the wild.

I've prepared this as two `git am`-able patches, inlined below:

1. **The fix + regression tests.** The zero-`v0` fallback turns out to be the only code path the existing operator tests never exercise (they all normalize a random `v0`), so this adds zero-`v0` tests covering all four sites (`partial_eigen` real + cplx, `partial_self_adjoint_eigen_imp`, `partial_svd_imp`). All pass; `cargo +nightly fmt` clean.
2. **A wasm32 CI job.** `.woodpecker/wasm.yaml` runs `cargo check --target wasm32-unknown-unknown --no-default-features --features linalg` (and `linalg,std`) — which alone would have caught this — plus a tiny `faer-wasm-test` crate (workspace-excluded, modeled on `faer-no-std-test`) that builds a **zero-import** wasm module and checks matmul + LU solve against hand-computed values under node 22. The `rayon` feature is deliberately not checked: `atomic-wait` has no wasm32 port, so `Par::Seq` is the wasm mode.

To apply: save the two blocks below as `0001.patch` / `0002.patch`, then

```
git checkout -b wasm32-32bit-fix
git am 0001.patch 0002.patch
```

I'm not set up on Codeberg, so I'm delivering these here — feel free to apply them directly, or tell me what you'd like changed.

<details>
<summary><b>0001 — fix 32-bit usize overflow in matrix-free operator modules (+ regression tests)</b></summary>

````text
From 0769dd8b676d1b9e0e4e7b2ce276eb4303c36ac2 Mon Sep 17 00:00:00 2001
From: andy-emerson <emerson.andrew@gmail.com>
Date: Tue, 7 Jul 2026 21:35:20 +0000
Subject: [PATCH 1/2] fix 32-bit usize overflow in matrix-free operator modules
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit

the zero-`v0` fallback in the partial eigen/svd operators splits
`n: usize` into two `u32` halves with `(n >> 32)`, which is a
compile-time overflow error on 32-bit targets:

    error: this arithmetic operation will overflow
      --> faer/src/./operator/eigen/mod.rs:319:13
        |
    319 |             let n1 = (n >> 32) as u32;
        |                      ^^^^^^^^^ attempt to shift right by `32_i32`, which would overflow

as a result faer 0.24.4 does not build at all on wasm32-unknown-unknown,
armv7, i686, or any other 32-bit target (with any feature set — the
operator module is unconditional). widen to u64 before shifting:
`((n as u64) >> 32) as u32`, which is a no-op on 64-bit and makes the
expression well-defined on 32-bit.

the fallback only runs when the caller passes an all-zero initial
vector, a path no existing test exercised; add zero-`v0` regression
tests for all four affected sites (partial_schur real + cplx,
partial_self_adjoint_eigen, partial_svd).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
---
 faer/src/operator/eigen/mod.rs              | 113 +++++++++++++++++++-
 faer/src/operator/self_adjoint_eigen/mod.rs |  52 ++++++++-
 faer/src/operator/svd/mod.rs                |  54 +++++++++-
 3 files changed, 215 insertions(+), 4 deletions(-)

diff --git a/faer/src/operator/eigen/mod.rs b/faer/src/operator/eigen/mod.rs
index bd253bf..eb9a998 100644
--- a/faer/src/operator/eigen/mod.rs
+++ b/faer/src/operator/eigen/mod.rs
@@ -316,7 +316,7 @@ fn partial_schur_real_imp<T: RealField>(
 				.for_each(|unzip!(y, x): Zip!(&mut T, &T)| *y = f * x);
 		} else {
 			let n0 = n as u32;
-			let n1 = (n >> 32) as u32;
+			let n1 = ((n as u64) >> 32) as u32;
 			let n = from_f64::<T>(n0 as f64) + from_f64::<T>(n1 as f64);
 			let f = n.sqrt().recip();
 			zip!(V.rb_mut().col_mut(0)).for_each(|unzip!(y)| *y = f.copy());
@@ -768,7 +768,7 @@ fn partial_schur_cplx_imp<T: ComplexField>(
 				.for_each(|unzip!(y, x): Zip!(&mut T, &T)| *y = x.mul_real(&f));
 		} else {
 			let n0 = n as u32;
-			let n1 = (n >> 32) as u32;
+			let n1 = ((n as u64) >> 32) as u32;
 			let n = from_f64::<T>(n0 as f64) + from_f64::<T>(n1 as f64);
 			let f = n.sqrt().recip();
 			zip!(V.rb_mut().col_mut(0)).for_each(|unzip!(y)| *y = f.copy());
@@ -1676,6 +1676,115 @@ mod tests {
 			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
 		}
 	}
+	// the zero-`v0` fallback splits `n: usize` into two `u32` halves with
+	// `n >> 32`, which fails to compile on 32-bit targets (wasm32, armv7,
+	// i686); it is also the only code path not covered by the other tests,
+	// which all pass a random nonzero `v0`
+	#[test]
+	fn test_zero_initial_vector_real() {
+		let rng = &mut StdRng::seed_from_u64(1);
+		let n = 100;
+		let n_eigval = 20;
+		let min_dim = 30;
+		let max_dim = 60;
+		let restarts = 1000;
+		let mat = CwiseMatDistribution {
+			nrows: n,
+			ncols: n,
+			dist: StandardNormal,
+		};
+		let A: Mat<f64> = mat.sample(rng);
+		let v0: Col<f64> = Col::zeros(n);
+		let A = A.as_ref();
+		let v0 = v0.as_ref();
+		let par = Par::Seq;
+		let mem = &mut MemBuffer::new(partial_eigen_scratch(
+			&A,
+			n_eigval,
+			par,
+			PartialEigenParams {
+				min_dim,
+				max_dim,
+				max_restarts: restarts,
+				..Default::default()
+			},
+		));
+		let mut V = Mat::zeros(n, n_eigval);
+		let mut w = vec![c64::ZERO; n_eigval];
+		let info = partial_eigen(
+			V.rb_mut(),
+			&mut w,
+			&A,
+			v0,
+			f64::EPSILON * 128.0,
+			par,
+			MemStack::new(mem),
+			PartialEigenParams {
+				min_dim,
+				max_dim,
+				max_restarts: restarts,
+				..Default::default()
+			},
+		);
+		assert!(info.n_converged_eigen > 0);
+		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));
+		let A = &zip!(A).map(|unzip!(x)| Complex::from(*x));
+		for j in 0..info.n_converged_eigen {
+			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
+		}
+	}
+	#[test]
+	fn test_zero_initial_vector_cplx() {
+		let rng = &mut StdRng::seed_from_u64(1);
+		let n = 100;
+		let n_eigval = 20;
+		let min_dim = 30;
+		let max_dim = 60;
+		let restarts = 1000;
+		let mat = CwiseMatDistribution {
+			nrows: n,
+			ncols: n,
+			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
+		};
+		let A: Mat<c64> = mat.sample(rng);
+		let v0: Col<c64> = Col::zeros(n);
+		let A = A.as_ref();
+		let v0 = v0.as_ref();
+		let par = Par::Seq;
+		let mem = &mut MemBuffer::new(partial_eigen_scratch(
+			&A,
+			n_eigval,
+			par,
+			PartialEigenParams {
+				min_dim,
+				max_dim,
+				max_restarts: restarts,
+				..Default::default()
+			},
+		));
+		let mut V = Mat::zeros(n, n_eigval);
+		let mut w = vec![c64::ZERO; n_eigval];
+		let info = partial_eigen(
+			V.rb_mut(),
+			&mut w,
+			&A,
+			v0,
+			f64::EPSILON * 128.0,
+			par,
+			MemStack::new(mem),
+			PartialEigenParams {
+				min_dim,
+				max_dim,
+				max_restarts: restarts,
+				..Default::default()
+			},
+		);
+		assert!(info.n_converged_eigen > 0);
+		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));
+		for j in 0..info.n_converged_eigen {
+			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
+		}
+	}
 	#[test]
 	fn test_small_cplx() {
 		let rng = &mut StdRng::seed_from_u64(1);
diff --git a/faer/src/operator/self_adjoint_eigen/mod.rs b/faer/src/operator/self_adjoint_eigen/mod.rs
index 48f4acf..7b36e1b 100644
--- a/faer/src/operator/self_adjoint_eigen/mod.rs
+++ b/faer/src/operator/self_adjoint_eigen/mod.rs
@@ -105,7 +105,7 @@ pub fn partial_self_adjoint_eigen_imp<T: ComplexField>(
 				.for_each(|unzip!(y, x): Zip!(&mut T, &T)| *y = x.mul_real(f));
 		} else {
 			let n0 = n as u32;
-			let n1 = (n >> 32) as u32;
+			let n1 = ((n as u64) >> 32) as u32;
 			let n = from_f64::<T>(n0 as f64) + from_f64::<T>(n1 as f64);
 			let f = n.sqrt().recip();
 			zip!(V.rb_mut().col_mut(0)).for_each(|unzip!(y)| *y = f.copy());
@@ -348,4 +348,54 @@ mod tests {
 			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
 		}
 	}
+
+	// the zero-`v0` fallback splits `n: usize` into two `u32` halves with
+	// `n >> 32`, which fails to compile on 32-bit targets (wasm32, armv7,
+	// i686); it is also the only code path not covered by the other tests,
+	// which all pass a random nonzero `v0`
+	#[test]
+	fn test_zero_initial_vector() {
+		let rng = &mut StdRng::seed_from_u64(1);
+		let n = 512;
+		let n_eigval = 32;
+		let mat = CwiseMatDistribution {
+			nrows: n,
+			ncols: n,
+			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
+		};
+		let A: Mat<c64> = mat.sample(rng);
+		let A = &A + A.adjoint();
+		let v0: Col<c64> = Col::zeros(n);
+		let A = A.as_ref();
+		let v0 = v0.as_ref();
+		let par = Par::Seq;
+		let mem = &mut MemBuffer::new(
+			crate::matrix_free::eigen::partial_eigen_scratch(
+				&A,
+				n_eigval,
+				par,
+				default(),
+			),
+		);
+		let mut V = Mat::zeros(n, n_eigval);
+		let mut w = vec![c64::ZERO; n_eigval];
+		let n_converged = partial_self_adjoint_eigen_imp(
+			V.rb_mut(),
+			&mut w,
+			&A,
+			v0,
+			32,
+			64,
+			n_eigval,
+			f64::EPSILON * 128.0,
+			1000,
+			par,
+			MemStack::new(mem),
+		);
+		assert!(w.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));
+		assert!(n_converged == n_eigval);
+		for j in 0..n_converged {
+			assert!((A * V.col(j) - Scale(w[j]) * V.col(j)).norm_l2() < 1e-10);
+		}
+	}
 }
diff --git a/faer/src/operator/svd/mod.rs b/faer/src/operator/svd/mod.rs
index 3a6aee9..2a064af 100644
--- a/faer/src/operator/svd/mod.rs
+++ b/faer/src/operator/svd/mod.rs
@@ -156,7 +156,7 @@ pub fn partial_svd_imp<T: ComplexField>(
 			.for_each(|unzip!(y, x): Zip!(&mut _, &_)| *y = x.mul_real(f));
 	} else {
 		let n0 = n as u32;
-		let n1 = (n >> 32) as u32;
+		let n1 = ((n as u64) >> 32) as u32;
 		let n = from_f64::<T>(n0 as f64) + from_f64::<T>(n1 as f64);
 		let ref f = n.sqrt().recip();
 		zip!(Q.rb_mut().col_mut(0)).for_each(|unzip!(y)| *y = f.copy());
@@ -415,4 +415,56 @@ mod tests {
 			);
 		}
 	}
+
+	// the zero-`v0` fallback splits `n: usize` into two `u32` halves with
+	// `n >> 32`, which fails to compile on 32-bit targets (wasm32, armv7,
+	// i686); it is also the only code path not covered by the other tests,
+	// which all pass a random nonzero `v0`
+	#[test]
+	fn test_zero_initial_vector() {
+		let rng = &mut StdRng::seed_from_u64(1);
+		let m = 768;
+		let n = 512;
+		let n_eigval = 32;
+		let mat = CwiseMatDistribution {
+			nrows: m,
+			ncols: n,
+			dist: ComplexDistribution::new(StandardNormal, StandardNormal),
+		};
+		let A: Mat<c64> = mat.sample(rng);
+		let v0: Col<c64> = Col::zeros(n);
+		let A = A.as_ref();
+		let v0 = v0.as_ref();
+		let par = Par::Seq;
+		let mem = &mut MemBuffer::new(StackReq::new::<u8>(1024 * 1024 * 512));
+		let mut U = Mat::zeros(m, n_eigval);
+		let mut V = Mat::zeros(n, n_eigval);
+		let mut s = vec![c64::ZERO; n_eigval];
+		let n_converged = partial_svd_imp(
+			U.rb_mut(),
+			V.rb_mut(),
+			&mut s,
+			&A,
+			v0,
+			32,
+			64,
+			n_eigval,
+			f64::EPSILON * 128.0,
+			1000,
+			par,
+			MemStack::new(mem),
+		);
+		assert!(s.iter().map(|x| x.norm()).is_sorted_by(|x, y| x >= y));
+		assert!(n_converged == n_eigval);
+		for j in 0..n_converged {
+			assert!(
+				(A.adjoint() * (A * V.col(j)) - Scale(s[j] * s[j]) * V.col(j))
+					.norm_l2() < 1e-10
+			);
+			assert!(
+				(A * (A.adjoint() * U.col(j)) - Scale(s[j] * s[j]) * U.col(j))
+					.norm_l2() < 1e-10
+			);
+		}
+	}
 }
-- 
2.43.0

````

</details>

<details>
<summary><b>0002 — add wasm32 CI job with a node smoke test</b></summary>

````text
From 58f15dfd06699bc92ab5f157d9cdb95ec5ed87e2 Mon Sep 17 00:00:00 2001
From: andy-emerson <emerson.andrew@gmail.com>
Date: Tue, 7 Jul 2026 21:37:59 +0000
Subject: [PATCH 2/2] add wasm32 CI job with a node smoke test

build-check faer for wasm32-unknown-unknown with
`--no-default-features --features linalg` (and `linalg,std`) so 32-bit
breakage like the `(n >> 32)` overflow can't silently regress, and add
`faer-wasm-test` (excluded from the workspace, like faer-no-std-test):
a `no_std` cdylib consumer that builds to a zero-import wasm module and
runs matmul + partial-pivot LU solve under node, checked against
hand-computed values.

there is real faer-on-wasm usage in the wild (#222); this makes the
target a checked configuration.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
---
 .woodpecker/wasm.yaml     | 19 +++++++++
 Cargo.toml                |  1 +
 faer-wasm-test/Cargo.toml | 18 +++++++++
 faer-wasm-test/run.mjs    | 22 ++++++++++
 faer-wasm-test/src/lib.rs | 85 +++++++++++++++++++++++++++++++++++++++
 5 files changed, 145 insertions(+)
 create mode 100644 .woodpecker/wasm.yaml
 create mode 100644 faer-wasm-test/Cargo.toml
 create mode 100644 faer-wasm-test/run.mjs
 create mode 100644 faer-wasm-test/src/lib.rs

diff --git a/.woodpecker/wasm.yaml b/.woodpecker/wasm.yaml
new file mode 100644
index 0000000..cc16a7c
--- /dev/null
+++ b/.woodpecker/wasm.yaml
@@ -0,0 +1,19 @@
+when:
+  - event: pull_request
+  - event: push
+    branch: main
+
+steps:
+  - name: build
+    image: rust
+    environment:
+      CARGO_TERM_COLOR: always
+    commands:
+      - rustup target add wasm32-unknown-unknown
+      - cd ./faer && cargo check --target wasm32-unknown-unknown --no-default-features --features linalg && cd ..
+      - cd ./faer && cargo check --target wasm32-unknown-unknown --no-default-features --features linalg,std && cd ..
+      - cd ./faer-wasm-test && cargo build --target wasm32-unknown-unknown --release
+  - name: smoke-test
+    image: node:22
+    commands:
+      - cd ./faer-wasm-test && node run.mjs
diff --git a/Cargo.toml b/Cargo.toml
index 2e259d5..4c2f047 100644
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -7,6 +7,7 @@ members = [
 ]
 exclude = [
   "faer-no-std-test",
+  "faer-wasm-test",
   "eigen-bench-setup"
 ]
 
diff --git a/faer-wasm-test/Cargo.toml b/faer-wasm-test/Cargo.toml
new file mode 100644
index 0000000..ccd7e97
--- /dev/null
+++ b/faer-wasm-test/Cargo.toml
@@ -0,0 +1,18 @@
+[package]
+name = "faer-wasm-test"
+version = "0.1.0"
+edition = "2021"
+publish = false
+
+[lib]
+crate-type = ["cdylib"]
+
+[dependencies]
+faer = { path = "../faer", default-features = false, features = ["linalg"] }
+
+[profile.release]
+opt-level = "z"
+lto = true
+codegen-units = 1
+panic = "abort"
+strip = true
diff --git a/faer-wasm-test/run.mjs b/faer-wasm-test/run.mjs
new file mode 100644
index 0000000..92dfa8e
--- /dev/null
+++ b/faer-wasm-test/run.mjs
@@ -0,0 +1,22 @@
+import { readFileSync } from 'node:fs';
+
+const wasm = readFileSync(
+	new URL(
+		'./target/wasm32-unknown-unknown/release/faer_wasm_test.wasm',
+		import.meta.url,
+	),
+);
+const { instance } = await WebAssembly.instantiate(wasm, {});
+const { matmul_trace, lu_solve_sum } = instance.exports;
+
+let failed = false;
+const check = (name, got, want) => {
+	const ok = Math.abs(got - want) < 1e-12;
+	console.log(`${name} = ${got} (want ${want}) ${ok ? 'ok' : 'FAIL'}`);
+	failed ||= !ok;
+};
+
+check('matmul_trace', matmul_trace(), 114);
+check('lu_solve_sum', lu_solve_sum(), 31 / 35);
+
+process.exit(failed ? 1 : 0);
diff --git a/faer-wasm-test/src/lib.rs b/faer-wasm-test/src/lib.rs
new file mode 100644
index 0000000..2b5daaf
--- /dev/null
+++ b/faer-wasm-test/src/lib.rs
@@ -0,0 +1,85 @@
+//! wasm32 smoke test: builds faer (`no_std`, `linalg` only) to
+//! `wasm32-unknown-unknown` as a zero-import module and exposes a couple of
+//! computations with hand-verified results, checked by `run.mjs` under node.
+
+#![no_std]
+
+extern crate alloc;
+
+use faer::Mat;
+
+/// trace of $A B$ for $A_{ij} = i + 2j + 1$, $B_{ij} = 3i + j - 2$ (3×3);
+/// hand-computed value: $21 + 36 + 57 = 114$
+#[no_mangle]
+pub extern "C" fn matmul_trace() -> f64 {
+	let a = Mat::from_fn(3, 3, |i, j| (i + 2 * j) as f64 + 1.0);
+	let b = Mat::from_fn(3, 3, |i, j| (3 * i + j) as f64 - 2.0);
+	let c = &a * &b;
+	c[(0, 0)] + c[(1, 1)] + c[(2, 2)]
+}
+
+/// sum of the solution of a 3×3 system via partial-pivot LU;
+/// hand-computed exact value: $31/35 = 0.8857142857\ldots$
+#[no_mangle]
+pub extern "C" fn lu_solve_sum() -> f64 {
+	use faer::prelude::*;
+	let a = faer::mat![[4.0, 3.0, 2.0], [2.0, 5.0, 1.0], [1.0, 1.0, 3.0f64]];
+	let rhs = faer::mat![[1.0], [2.0], [3.0f64]];
+	let lu = a.partial_piv_lu();
+	let x = lu.solve(&rhs);
+	x[(0, 0)] + x[(1, 0)] + x[(2, 0)]
+}
+
+// leak-only bump allocator over `memory.grow`, so the module needs no
+// imports at all; fine for a smoke test
+mod wasm_shim {
+	use core::alloc::{GlobalAlloc, Layout};
+
+	struct Bump;
+
+	unsafe extern "C" {
+		static __heap_base: u8;
+	}
+
+	static mut OFFSET: usize = 0;
+
+	#[inline]
+	unsafe fn offset() -> usize {
+		unsafe {
+			if OFFSET == 0 {
+				OFFSET = &__heap_base as *const u8 as usize;
+			}
+			OFFSET
+		}
+	}
+
+	unsafe impl GlobalAlloc for Bump {
+		unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
+			unsafe {
+				let align = layout.align().max(16);
+				let base = (offset() + align - 1) & !(align - 1);
+				let end = base + layout.size();
+				let cur_pages = core::arch::wasm32::memory_size(0);
+				let need = (end + 0xFFFF) / 0x10000;
+				if need > cur_pages
+					&& core::arch::wasm32::memory_grow(0, need - cur_pages)
+						== usize::MAX
+				{
+					return core::ptr::null_mut();
+				}
+				OFFSET = end;
+				base as *mut u8
+			}
+		}
+
+		unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
+	}
+
+	#[global_allocator]
+	static A: Bump = Bump;
+
+	#[panic_handler]
+	fn panic(_: &core::panic::PanicInfo) -> ! {
+		core::arch::wasm32::unreachable()
+	}
+}
-- 
2.43.0

````

</details>

**Side observation** (deliberately not changed here): the split converts `n` to the scalar type as `from_f64(n0 as f64) + from_f64(n1 as f64)` — the high half is never scaled by 2^32, so the sum equals `n` only for `n < 2^32`. If exact conversion is the intent, the `n1` term needs a `* 4294967296.0`; happy to fold that in if wanted.
