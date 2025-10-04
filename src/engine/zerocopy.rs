// zerocopy.rs 

use std::io::{self, Error, Result};
use std::os::raw::{c_int, c_void};



#[cfg(target_os = "windows")]
#[link(name = "zerocopy", kind = "static")]
extern "C" {
    fn zc_transfer_win(
        out_sock: usize,
        in_file: usize,
        offset_ptr: *mut i64,
        count: usize,
    ) -> isize;

    fn zc_transfer_all_win(
        out_sock: usize,
        in_file: usize,
        offset_ptr: *mut i64,
        count: usize,
    ) -> isize;
}

#[cfg(unix)]
#[link(name = "zerocopy", kind = "static")]
extern "C" {
    fn zc_transfer(
        out_fd: c_int,
        in_fd: c_int,
        offset_ptr: *mut i64,
        count: usize,
    ) -> isize;

    fn zc_transfer_all(
        out_fd: c_int,
        in_fd: c_int,
        offset_ptr: *mut i64,
        count: usize,
    ) -> isize;
}

//safe rust wrappers
pub fn transfer(src: usize, dst: usize, offset: Option<&mut i64>, count: usize) -> Result<usize> {
    let mut off_ptr = offset.map(|r| r as  *mut i64).unwrap_or(std::ptr::null_mut());

    let ret = unsafe {
        #[cfg(target_os = "windows")]
        {
            zc_transfer_win(dst, src, off_ptr, count)
        }
        #[cfg(unix)]
        {
            zc_transfer(dst as c_int, src as c_int, off_ptr, count)
        }
    };

    if ret < 0 {
        Err(Error::last_os_error())
    } else {
        Ok(ret as usize)
    }
}

/// Transfer all data until EOF (using platform-specific zero-copy primitives)
pub fn transfer_all(src: usize, dst: usize, offset: Option<&mut i64>, count: usize) -> Result<usize> {
    let mut off_ptr = offset.map(|r| r as *mut i64).unwrap_or(std::ptr::null_mut());

    let ret = unsafe {
        #[cfg(target_os = "windows")]
        {
            zc_transfer_all_win(dst, src, off_ptr, count)
        }
        #[cfg(unix)]
        {
            zc_transfer_all(dst as c_int, src as c_int, off_ptr, count)
        }
    };

    if ret < 0 {
        Err(Error::last_os_error())
    } else {
        Ok(ret as usize)
    }
}



#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(target_os = "windows")]
use std::os::windows::io::AsRawHandle;


pub fn transfer_files(src: &std::fs::File, dst: &std::fs::File, count: usize) -> Result<usize> {
    #[cfg(unix)]
    {
        transfer(src.as_raw_fd() as usize, dst.as_raw_fd() as usize, None, count)
    }
    #[cfg(target_os = "windows")]
    {
        transfer(src.as_raw_handle() as usize, dst.as_raw_handle() as usize, None, count)
    }
}

/// Verify if zerocopy linkage works correctly
pub fn test_linkage() -> bool {
    let test = unsafe {
        #[cfg(target_os = "windows")]
        {
            zc_transfer_win(0, 0, std::ptr::null_mut(), 0)
        }
        #[cfg(unix)]
        {
            zc_transfer(0, 0, std::ptr::null_mut(), 0)
        }
    };
    test == 0
}


