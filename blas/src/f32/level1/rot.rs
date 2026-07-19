//! `rot` — apply a plane rotation: (xᵢ, yᵢ) ← (c·xᵢ + s·yᵢ, c·yᵢ − s·xᵢ).
//!
//! Implementation: elementwise stream (4 lanes, 2× unrolled).
//!
//! Rounding contract: each output is two multiply roundings then one
//! add/sub rounding, in that order — bit-identical to the scalar
//! definition on every target. A fused variant is a future per-op
//! measurement.

use crate::f32::lanes::F32x4;

/// Apply the rotation with cosine `c` and sine `s` to the vector pair.
/// Panics on length mismatch.
pub fn rot(x: &mut [f32], y: &mut [f32], c: f32, s: f32) {
	assert_eq!(x.len(), y.len(), "rot: length mismatch");
	unsafe { imp(x.as_mut_ptr(), y.as_mut_ptr(), c, s, x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *mut f32, yp: *mut f32, c: f32, s: f32, len: usize) {
	let vc = F32x4::splat(c);
	let vs = F32x4::splat(s);
	let mut i = 0usize;
	while i + 8 <= len {
		let x0 = F32x4::load(xp.add(i));
		let y0 = F32x4::load(yp.add(i));
		let x1 = F32x4::load(xp.add(i + 4));
		let y1 = F32x4::load(yp.add(i + 4));
		x0.mul(vc).add(y0.mul(vs)).store(xp.add(i));
		y0.mul(vc).sub(x0.mul(vs)).store(yp.add(i));
		x1.mul(vc).add(y1.mul(vs)).store(xp.add(i + 4));
		y1.mul(vc).sub(x1.mul(vs)).store(yp.add(i + 4));
		i += 8;
	}
	while i < len {
		let xi = *xp.add(i);
		let yi = *yp.add(i);
		*xp.add(i) = xi * c + yi * s;
		*yp.add(i) = yi * c - xi * s;
		i += 1;
	}
}
