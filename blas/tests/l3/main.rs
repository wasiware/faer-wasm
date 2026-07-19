//! Level 3 correctness per the testing contract (../../README.md),
//! one test file per BLAS routine mirroring src/l3/ (naming
//! convention: src/l1/README.md). Shared helpers in common.rs.

mod common;

mod dgemm;
mod dsymm;
mod dsyr2k;
mod dsyrk;
mod dtrmm;
mod dtrsm;
mod sgemm;
mod ssymm;
mod ssyr2k;
mod ssyrk;
mod strmm;
mod strsm;
