//! `sdot` — sdot product: xᵀy.
//!
//! Implementation: reduction stream (4 accumulator registers = 16 lanes,
//! folded in a fixed order at the end).
//!
//! Rounding contract: lane-parallel accumulation reorders the additions
//! relative to a sequential loop — a legitimately different, equally
//! valid rounding sequence. Tested against a compensated-summation
//! reference within n-scaled error bounds; native ↔ wasm bit-identical
//! by the lane-emulation construction (see `lanes.rs`).

use crate::lanes::F32x4;

/// Returns xᵀy. Panics on length mismatch.
pub fn sdot(x: &[f32], y: &[f32]) -> f32 {
	assert_eq!(x.len(), y.len(), "sdot: length mismatch");
	unsafe { imp(x.as_ptr(), y.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *const f32, yp: *const f32, len: usize) -> f32 {
	let mut acc0 = F32x4::splat(0.0);
	let mut acc1 = F32x4::splat(0.0);
	let mut acc2 = F32x4::splat(0.0);
	let mut acc3 = F32x4::splat(0.0);
	let mut i = 0usize;
	// 4 accumulator registers (tuning lever, 2026-07-19)
	while i + 16 <= len {
		acc0 = acc0.add(F32x4::load(xp.add(i)).mul(F32x4::load(yp.add(i))));
		acc1 = acc1.add(F32x4::load(xp.add(i + 4)).mul(F32x4::load(yp.add(i + 4))));
		acc2 = acc2.add(F32x4::load(xp.add(i + 8)).mul(F32x4::load(yp.add(i + 8))));
		acc3 = acc3.add(F32x4::load(xp.add(i + 12)).mul(F32x4::load(yp.add(i + 12))));
		i += 16;
	}
	let acc = acc0.add(acc1).add(acc2.add(acc3));
	let mut s = (acc.lane0() + acc.lane1()) + (acc.lane2() + acc.lane3());
	while i < len {
		s += *xp.add(i) * *yp.add(i);
		i += 1;
	}
	s
}
