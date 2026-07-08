#![cfg_attr(target_arch = "wasm32", no_std)]

extern crate alloc;

use faer::Mat;

// stage "matmul": just matrix multiply
// stage "lu": + partial-pivot LU solve
// stage "full": + QR, SVD, self-adjoint EVD, general EVD

#[no_mangle]
pub extern "C" fn matmul_trace() -> f64 {
    let a = Mat::from_fn(3, 3, |i, j| (i + 2 * j) as f64 + 1.0);
    let b = Mat::from_fn(3, 3, |i, j| (3 * i + j) as f64 - 2.0);
    let c = &a * &b;
    c[(0, 0)] + c[(1, 1)] + c[(2, 2)]
}

#[cfg(any(feature = "lu", feature = "full"))]
#[no_mangle]
pub extern "C" fn lu_solve_sum() -> f64 {
    use faer::prelude::*;
    let a = faer::mat![[4.0, 3.0, 2.0], [2.0, 5.0, 1.0], [1.0, 1.0, 3.0f64]];
    let rhs = faer::mat![[1.0], [2.0], [3.0f64]];
    let lu = a.partial_piv_lu();
    let x = lu.solve(&rhs);
    x[(0, 0)] + x[(1, 0)] + x[(2, 0)]
}

#[cfg(feature = "full")]
#[no_mangle]
pub extern "C" fn qr_svd_evd_probe() -> f64 {
    use faer::prelude::*;
    let a = faer::mat![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0f64]];
    let qr = a.qr();
    let r00 = qr.R()[(0, 0)];
    let svd = a.svd().unwrap();
    let s0 = svd.S()[0];
    let evd = a.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let e0 = evd.S()[0];
    // general (non-hermitian) eigendecomposition
    let g = faer::mat![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0f64]];
    let ge: alloc::vec::Vec<faer::c64> = g.eigenvalues().unwrap();
    r00 + s0 + e0 + ge[0].re
}

#[cfg(target_arch = "wasm32")]
mod wasm_shim {
    use core::alloc::{GlobalAlloc, Layout};

    // leak-only bump allocator over memory.grow, so the module needs no
    // imports at all; fine for a smoke test
    struct Bump;

    extern "C" {
        static __heap_base: u8;
    }

    static mut OFFSET: usize = 0;

    #[inline]
    unsafe fn offset() -> usize {
        if OFFSET == 0 {
            OFFSET = &__heap_base as *const u8 as usize;
        }
        OFFSET
    }

    unsafe impl GlobalAlloc for Bump {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let align = layout.align().max(16);
            let base = (offset() + align - 1) & !(align - 1);
            let end = base + layout.size();
            let cur_pages = core::arch::wasm32::memory_size(0);
            let need = (end + 0xffff) / 0x10000;
            if need > cur_pages {
                if core::arch::wasm32::memory_grow(0, need - cur_pages) == usize::MAX {
                    return core::ptr::null_mut();
                }
            }
            OFFSET = end;
            base as *mut u8
        }
        unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
    }

    #[global_allocator]
    static A: Bump = Bump;

    #[panic_handler]
    fn panic(_: &core::panic::PanicInfo) -> ! {
        core::arch::wasm32::unreachable()
    }
}
