//! `sasum` — sum of absolute values (ℓ¹ norm): Σ|xᵢ|.
//!
//! Implementation: reduction stream (4 accumulator registers = 16 lanes,
//! `abs` in-lane, folded in a fixed order at the end). Shape ported
//! from the raced f64 variant (3.5–4× over the plain loop, three
//! runner draws, docs/blas-ab-2026-07.md step 2 — the plain loop's
//! serial dependent-add chain is latency-bound, not bandwidth-bound);
//! f32 measurements: step 10.
//!
//! Rounding contract: lane-parallel accumulation reorders the additions
//! — tested against a compensated-summation reference within n-scaled
//! error bounds; native ↔ wasm bit-identical by construction.

use crate::lanes::F32x4;

/// Returns Σ|xᵢ|.
pub fn sasum(x: &[f32]) -> f32 {
	unsafe { imp(x.as_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(xp: *const f32, len: usize) -> f32 {
	let mut a0 = F32x4::splat(0.0);
	let mut a1 = F32x4::splat(0.0);
	let mut a2 = F32x4::splat(0.0);
	let mut a3 = F32x4::splat(0.0);
	let mut i = 0usize;
	// 4 accumulator registers (tuning lever, 2026-07-19): the 2-register
	// version left the reductions latency-bound at 60-80%% of triad
	while i + 16 <= len {
		a0 = a0.add(F32x4::load(xp.add(i)).abs());
		a1 = a1.add(F32x4::load(xp.add(i + 4)).abs());
		a2 = a2.add(F32x4::load(xp.add(i + 8)).abs());
		a3 = a3.add(F32x4::load(xp.add(i + 12)).abs());
		i += 16;
	}
	let a = a0.add(a1).add(a2.add(a3));
	let mut s = (a.lane0() + a.lane1()) + (a.lane2() + a.lane3());
	while i < len {
		s += (*xp.add(i)).abs();
		i += 1;
	}
	s
}
