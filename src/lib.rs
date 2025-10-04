
pub mod engine {
    pub mod zerocopy;
}

use std::io::Result;
use std::fs::File;

/// Core FRD-PU API layer
/// Provides high-level access to zerocopy streaming engine.
pub struct FRDPU;

impl FRDPU {
    /// Perform a single zero-copy transfer between two file descriptors
    pub fn transfer_fd(src_fd: usize, dst_fd: usize, count: usize) -> Result<usize> {
        engine::zerocopy::transfer(src_fd, dst_fd, None, count)
    }

    /// Perform a full zero-copy transfer between two file descriptors until EOF
    pub fn transfer_all_fd(src_fd: usize, dst_fd: usize, count: usize) -> Result<usize> {
        engine::zerocopy::transfer_all(src_fd, dst_fd, None, count)
    }

    /// Perform a zero-copy transfer between two standard File objects
    pub fn transfer_files(src: &File, dst: &File, count: usize) -> Result<usize> {
        engine::zerocopy::transfer_files(src, dst, count)
    }

    /// Verify zero-copy linkage works
    pub fn test_linkage() -> bool {
        engine::zerocopy::test_linkage()
    }
}


// Example usage (for docs/testing)

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_zerocopy_link() {
//         assert!(FRDPU::test_linkage());
//     }
//
//     #[test]
//     fn test_transfer() {
//         let src = std::fs::File::open("Cargo.toml").unwrap();
//         let dst = std::fs::File::create("Cargo_copy.toml").unwrap();
//         let transferred = FRDPU::transfer_files(&src, &dst, 4096).unwrap();
//         println!("Transferred {} bytes", transferred);
//     }
// }
