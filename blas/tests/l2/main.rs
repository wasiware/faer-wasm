//! Level 2 correctness per the testing contract (../../README.md),
//! one test file per BLAS routine mirroring src/l2/ (naming
//! convention: src/l1/README.md). Shared helpers in common.rs.

mod common;

mod dgemv;
mod dger;
mod dsymv;
mod dsyr;
mod dsyr2;
mod dtrmv;
mod dtrsv;
mod sgemv;
mod sger;
mod ssymv;
mod ssyr;
mod ssyr2;
mod strmv;
mod strsv;
