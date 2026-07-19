//! `sswap` — exchange two vectors: x ↔ y.
//!
//! Implementation: elementwise stream (4 lanes, 2× unrolled). Shape
//! ported from the raced f64 variant (SIMD beat the auto-vectorized
//! plain loop 1.15–1.33×, three runner draws, docs/blas-ab-2026-07.md
//! step 2); f32 measurements: step 10.
//!
//! Rounding contract: none — bytes move unchanged, bit-for-bit.

use crate::lanes::F32x4;

/// x ↔ y. Panics on length mismatch.
pub fn sswap(x: &mut [f32], y: &mut [f32]) {
	assert_eq!(x.len(), y.len(), "sswap: length mismatch");
	unsafe { imp(x.as_mut_ptr(), y.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *mut f32, yp: *mut f32, len: usize) {
	let mut i = 0usize;
	while i + 8 <= len {
		let x0 = F32x4::load(xp.add(i));
		let y0 = F32x4::load(yp.add(i));
		let x1 = F32x4::load(xp.add(i + 4));
		let y1 = F32x4::load(yp.add(i + 4));
		y0.store(xp.add(i));
		x0.store(yp.add(i));
		y1.store(xp.add(i + 4));
		x1.store(yp.add(i + 4));
		i += 8;
	}
	while i < len {
		let t = *xp.add(i);
		*xp.add(i) = *yp.add(i);
		*yp.add(i) = t;
		i += 1;
	}
}
