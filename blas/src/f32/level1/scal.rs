//! `scal` — scalar × vector: x ← αx.
//!
//! Implementation: elementwise stream (4 lanes, 2× unrolled).
//!
//! Rounding contract: one multiply rounding per element — bit-identical
//! to the scalar definition on every target.

use crate::f32::lanes::F32x4;

/// x ← αx.
pub fn scal(alpha: f32, x: &mut [f32]) {
	unsafe { imp(alpha, x.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(alpha: f32, xp: *mut f32, len: usize) {
	let va = F32x4::splat(alpha);
	let mut i = 0usize;
	while i + 8 <= len {
		let x0 = F32x4::load(xp.add(i));
		let x1 = F32x4::load(xp.add(i + 4));
		x0.mul(va).store(xp.add(i));
		x1.mul(va).store(xp.add(i + 4));
		i += 8;
	}
	while i < len {
		*xp.add(i) *= alpha;
		i += 1;
	}
}
