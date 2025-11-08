pub mod engine {
    pub mod zerocopy;
}
use std::io::Result;
use std::fs::File;
pub struct FRDPU;
impl FRDPU {
    pub fn transfer_fd(src_fd: usize, dst_fd: usize, count: usize) -> Result<usize> {
        engine::zerocopy::transfer(src_fd, dst_fd, None, count)
    }
    pub fn transfer_all_fd(src_fd: usize, dst_fd: usize, count: usize) -> Result<usize> {
        engine::zerocopy::transfer_all(src_fd, dst_fd, None, count)
    }
    pub fn transfer_files(src: &File, dst: &File, count: usize) -> Result<usize> {
        engine::zerocopy::transfer_files(src, dst, count)
    }
    pub fn test_linkage() -> bool {
        engine::zerocopy::test_linkage()
    }
}
