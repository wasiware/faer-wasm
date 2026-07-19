//! Four f32 SIMD lanes: wasm simd128 `v128` on wasm32, a bit-identical
//! four-element emulation everywhere else — the f32 twin of the f64
//! layer's two-lane type (see `crate::lanes` for the full rationale).
//! Reductions built on this fold their accumulator lanes in a fixed
//! order, so native and wasm produce the same bits by construction.
//!
//! Every method is `unsafe` and, on wasm, carries
//! `#[target_feature(enable = "simd128")]`: simd128 is NOT in rustc's
//! default wasm32 feature set (measured 2026-07-18 — an unannotated
//! wrapper compiled every lane op as an out-of-line call and ran the
//! reductions 6× slower), so the feature must be enabled on the whole
//! call chain for the intrinsics to inline.

#[cfg(target_arch = "wasm32")]
mod imp {
	use core::arch::wasm32::*;

	#[derive(Clone, Copy)]
	pub struct F32x4(v128);

	#[allow(clippy::missing_safety_doc)]
	impl F32x4 {
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn splat(v: f32) -> Self {
			Self(f32x4_splat(v))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn load(p: *const f32) -> Self {
			Self(v128_load(p as *const v128))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn store(self, p: *mut f32) {
			v128_store(p as *mut v128, self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn add(self, o: Self) -> Self {
			Self(f32x4_add(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn sub(self, o: Self) -> Self {
			Self(f32x4_sub(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn mul(self, o: Self) -> Self {
			Self(f32x4_mul(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn div(self, o: Self) -> Self {
			Self(f32x4_div(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn abs(self) -> Self {
			Self(f32x4_abs(self.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn pmax(self, o: Self) -> Self {
			Self(f32x4_pmax(self.0, o.0))
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane0(self) -> f32 {
			f32x4_extract_lane::<0>(self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane1(self) -> f32 {
			f32x4_extract_lane::<1>(self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane2(self) -> f32 {
			f32x4_extract_lane::<2>(self.0)
		}
		#[inline]
		#[target_feature(enable = "simd128")]
		pub unsafe fn lane3(self) -> f32 {
			f32x4_extract_lane::<3>(self.0)
		}
	}
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
	#[derive(Clone, Copy)]
	pub struct F32x4([f32; 4]);

	// `unsafe fn` for signature parity with the wasm impl — the f32
	// streams call these inside one feature-annotated unsafe inner fn.
	#[allow(clippy::missing_safety_doc)]
	impl F32x4 {
		#[inline(always)]
		pub unsafe fn splat(v: f32) -> Self {
			Self([v; 4])
		}
		#[inline(always)]
		pub unsafe fn load(p: *const f32) -> Self {
			Self([*p, *p.add(1), *p.add(2), *p.add(3)])
		}
		#[inline(always)]
		pub unsafe fn store(self, p: *mut f32) {
			*p = self.0[0];
			*p.add(1) = self.0[1];
			*p.add(2) = self.0[2];
			*p.add(3) = self.0[3];
		}
		#[inline(always)]
		pub unsafe fn add(self, o: Self) -> Self {
			Self(core::array::from_fn(|i| self.0[i] + o.0[i]))
		}
		#[inline(always)]
		pub unsafe fn sub(self, o: Self) -> Self {
			Self(core::array::from_fn(|i| self.0[i] - o.0[i]))
		}
		#[inline(always)]
		pub unsafe fn mul(self, o: Self) -> Self {
			Self(core::array::from_fn(|i| self.0[i] * o.0[i]))
		}
		#[inline(always)]
		pub unsafe fn div(self, o: Self) -> Self {
			Self(core::array::from_fn(|i| self.0[i] / o.0[i]))
		}
		#[inline(always)]
		pub unsafe fn abs(self) -> Self {
			Self(core::array::from_fn(|i| self.0[i].abs()))
		}
		// wasm f32x4_pmax is lane-wise `a < b ? b : a` (NOT
		// NaN-propagating like fmax) — emulated with exactly that
		// comparison.
		#[inline(always)]
		pub unsafe fn pmax(self, o: Self) -> Self {
			#[inline(always)]
			fn pm(a: f32, b: f32) -> f32 {
				if a < b {
					b
				} else {
					a
				}
			}
			Self(core::array::from_fn(|i| pm(self.0[i], o.0[i])))
		}
		#[inline(always)]
		pub unsafe fn lane0(self) -> f32 {
			self.0[0]
		}
		#[inline(always)]
		pub unsafe fn lane1(self) -> f32 {
			self.0[1]
		}
		#[inline(always)]
		pub unsafe fn lane2(self) -> f32 {
			self.0[2]
		}
		#[inline(always)]
		pub unsafe fn lane3(self) -> f32 {
			self.0[3]
		}
	}
}

pub(crate) use imp::F32x4;
