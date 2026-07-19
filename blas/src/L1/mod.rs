//! Level 1: vector operations, one file per BLAS routine (netlib
//! naming — see README.md in this folder for the convention). The
//! d-prefixed routines are f64, s-prefixed f32; i-prefixed index
//! routines carry the type as their second letter (idamax/isamax).
//! `drotg`/`srotg` also export their `Givens` result structs via
//! their module paths.

pub mod dasum;
pub mod daxpy;
pub mod dcopy;
pub mod ddot;
pub mod dnrm2;
pub mod drot;
pub mod drotg;
pub mod dscal;
pub mod dswap;
pub mod idamax;
pub mod isamax;
pub mod sasum;
pub mod saxpy;
pub mod scopy;
pub mod sdot;
pub mod snrm2;
pub mod srot;
pub mod srotg;
pub mod sscal;
pub mod sswap;

pub use dasum::dasum;
pub use daxpy::daxpy;
pub use dcopy::dcopy;
pub use ddot::ddot;
pub use dnrm2::dnrm2;
pub use drot::drot;
pub use drotg::drotg;
pub use dscal::dscal;
pub use dswap::dswap;
pub use idamax::idamax;
pub use isamax::isamax;
pub use sasum::sasum;
pub use saxpy::saxpy;
pub use scopy::scopy;
pub use sdot::sdot;
pub use snrm2::snrm2;
pub use srot::srot;
pub use srotg::srotg;
pub use sscal::sscal;
pub use sswap::sswap;
