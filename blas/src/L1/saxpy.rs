//! `saxpy` — scaled vector addition: y ← αx + y.
//!
//! Implementation: elementwise stream (4 lanes, 2× unrolled).
//!
//! Rounding contract: each element is `y[i] + α·x[i]` — one multiply
//! rounding, one add rounding, bit-identical to the scalar definition
//! on every target. An FMA variant (single rounding) is a future
//! per-op measurement, not built yet.

use crate::lanes::F32x4;

/// y ← αx + y. Panics on length mismatch.
pub fn saxpy(alpha: f32, x: &[f32], y: &mut [f32]) {
	assert_eq!(x.len(), y.len(), "saxpy: length mismatch");
	unsafe { imp(alpha, x.as_ptr(), y.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(alpha: f32, xp: *const f32, yp: *mut f32, len: usize) {
	let va = F32x4::splat(alpha);
	let mut i = 0usize;
	while i + 8 <= len {
		let x0 = F32x4::load(xp.add(i));
		let y0 = F32x4::load(yp.add(i));
		let x1 = F32x4::load(xp.add(i + 4));
		let y1 = F32x4::load(yp.add(i + 4));
		y0.add(x0.mul(va)).store(yp.add(i));
		y1.add(x1.mul(va)).store(yp.add(i + 4));
		i += 8;
	}
	while i < len {
		*yp.add(i) += *xp.add(i) * alpha;
		i += 1;
	}
}
