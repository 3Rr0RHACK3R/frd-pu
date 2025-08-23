// src/buffer_pool.rs

//! # Buffer Pool Module
//!
//! A high-performance, zero-dependency buffer pool implementation designed for extreme efficiency.
//! This module provides reusable I/O buffers that eliminate constant allocation/deallocation cycles,
//! making it perfect for high-throughput data processing and streaming applications.
//!
//! ## Key Features:
//! * **Zero-allocation reuse** - Buffers are recycled instead of deallocated
//! * **Multiple pool sizes** - Different buffer sizes for different use cases
//! * **Thread-safe operations** - Safe to use across multiple threads
//! * **Automatic growth** - Pools can grow when under pressure
//! * **Memory-efficient** - Returns memory to system when pools shrink
//! * **Fast access** - O(1) buffer acquisition and release

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use std::fmt;

/// Errors that can occur during buffer pool operations
#[derive(Debug, Clone)]
pub enum BufferPoolError {
    /// Pool is at maximum capacity and cannot grow
    PoolExhausted,
    /// Invalid buffer size requested
    InvalidSize,
    /// Buffer was not allocated from this pool
    InvalidBuffer,
    /// Pool configuration error
    ConfigError(String),
}

impl fmt::Display for BufferPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoolExhausted => write!(f, "Buffer pool is exhausted"),
            Self::InvalidSize => write!(f, "Invalid buffer size"),
            Self::InvalidBuffer => write!(f, "Buffer not from this pool"),
            Self::ConfigError(msg) => write!(f, "Pool config error: {}", msg),
        }
    }
}

impl std::error::Error for BufferPoolError {}

/// Statistics for monitoring buffer pool performance
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    /// Total number of buffers allocated
    pub total_allocated: usize,
    /// Number of buffers currently available in pool
    pub available_buffers: usize,
    /// Number of buffers currently in use
    pub buffers_in_use: usize,
    /// Peak number of buffers allocated simultaneously
    pub peak_allocation: usize,
    /// Total number of buffer acquisitions
    pub total_acquisitions: usize,
    /// Total number of buffer releases
    pub total_releases: usize,
    /// Number of times pool had to grow
    pub growth_events: usize,
}

impl BufferPoolStats {
    fn new() -> Self {
        Self {
            total_allocated: 0,
            available_buffers: 0,
            buffers_in_use: 0,
            peak_allocation: 0,
            total_acquisitions: 0,
            total_releases: 0,
            growth_events: 0,
        }
    }
}

/// A pooled buffer that automatically returns to the pool when dropped
pub struct PooledBuffer {
    buffer: Vec<u8>,
    pool: Arc<Mutex<BufferPoolInner>>,
    buffer_size: usize,
}

impl PooledBuffer {
    /// Get the buffer as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer
    }
    
    /// Get the buffer as an immutable slice
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }
    
    /// Get the capacity of this buffer
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }
    
    /// Clear the buffer contents (sets length to 0)
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
    
    /// Resize the buffer to a specific length, filling with zeros if needed
    pub fn resize(&mut self, new_len: usize) {
        self.buffer.resize(new_len, 0);
    }
    
    /// Get the current length of data in the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
    
    /// Get mutable access to the underlying Vec<u8>
    pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        &mut self.buffer
    }
    
    /// Get immutable access to the underlying Vec<u8>
    pub fn as_vec(&self) -> &Vec<u8> {
        &self.buffer
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return buffer to pool when dropped
        let mut buffer = std::mem::take(&mut self.buffer);
        buffer.clear(); // Clear contents for reuse
        
        if let Ok(mut pool) = self.pool.lock() {
            pool.return_buffer(buffer, self.buffer_size);
        }
    }
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}

impl AsMut<[u8]> for PooledBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }
}

/// Internal buffer pool structure
struct BufferPoolInner {
    /// Pools for different buffer sizes
    pools: std::collections::HashMap<usize, VecDeque<Vec<u8>>>,
    /// Maximum number of buffers per size
    max_buffers_per_size: usize,
    /// Statistics
    stats: BufferPoolStats,
}

impl BufferPoolInner {
    fn new(max_buffers_per_size: usize) -> Self {
        Self {
            pools: std::collections::HashMap::new(),
            max_buffers_per_size,
            stats: BufferPoolStats::new(),
        }
    }
    
    fn get_buffer(&mut self, size: usize) -> Vec<u8> {
        self.stats.total_acquisitions += 1;
        
        let pool = self.pools.entry(size).or_insert_with(VecDeque::new);
        
        if let Some(mut buffer) = pool.pop_front() {
            // Ensure buffer has the right capacity
            if buffer.capacity() < size {
                buffer.reserve(size - buffer.capacity());
            }
            buffer.resize(size, 0);
            self.stats.available_buffers = self.stats.available_buffers.saturating_sub(1);
            self.stats.buffers_in_use += 1;
            buffer
        } else {
            // Create new buffer
            let buffer = vec![0u8; size];
            self.stats.total_allocated += 1;
            self.stats.buffers_in_use += 1;
            self.stats.peak_allocation = self.stats.peak_allocation.max(self.stats.buffers_in_use);
            buffer
        }
    }
    
    fn return_buffer(&mut self, buffer: Vec<u8>, size: usize) {
        self.stats.total_releases += 1;
        self.stats.buffers_in_use = self.stats.buffers_in_use.saturating_sub(1);
        
        let pool = self.pools.entry(size).or_insert_with(VecDeque::new);
        
        if pool.len() < self.max_buffers_per_size {
            pool.push_back(buffer);
            self.stats.available_buffers += 1;
        }
        // If pool is full, let buffer drop (deallocate)
    }
}

/// A high-performance buffer pool for zero-allocation buffer reuse
pub struct BufferPool {
    inner: Arc<Mutex<BufferPoolInner>>,
}

impl BufferPool {
    /// Create a new buffer pool with default settings
    pub fn new() -> Self {
        Self::with_max_buffers(64) // Default: up to 64 buffers per size
    }
    
    /// Create a new buffer pool with custom maximum buffers per size
    pub fn with_max_buffers(max_buffers_per_size: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BufferPoolInner::new(max_buffers_per_size))),
        }
    }
    
    /// Get a buffer of the specified size
    pub fn get_buffer(&self, size: usize) -> Result<PooledBuffer, BufferPoolError> {
        if size == 0 {
            return Err(BufferPoolError::InvalidSize);
        }
        
        let buffer = {
            let mut inner = self.inner.lock().map_err(|_| {
                BufferPoolError::ConfigError("Failed to acquire pool lock".to_string())
            })?;
            inner.get_buffer(size)
        };
        
        Ok(PooledBuffer {
            buffer,
            pool: Arc::clone(&self.inner),
            buffer_size: size,
        })
    }
    
    /// Get current pool statistics
    pub fn stats(&self) -> Result<BufferPoolStats, BufferPoolError> {
        let inner = self.inner.lock().map_err(|_| {
            BufferPoolError::ConfigError("Failed to acquire pool lock".to_string())
        })?;
        Ok(inner.stats.clone())
    }
    
    /// Clear all buffers from the pool (forces deallocation)
    pub fn clear(&self) -> Result<(), BufferPoolError> {
        let mut inner = self.inner.lock().map_err(|_| {
            BufferPoolError::ConfigError("Failed to acquire pool lock".to_string())
        })?;
        
        inner.pools.clear();
        inner.stats.available_buffers = 0;
        Ok(())
    }
    
    /// Shrink pools to remove excess buffers (keeps only half)
    pub fn shrink(&self) -> Result<(), BufferPoolError> {
        let mut inner = self.inner.lock().map_err(|_| {
            BufferPoolError::ConfigError("Failed to acquire pool lock".to_string())
        })?;
        
        let mut removed = 0;
        for pool in inner.pools.values_mut() {
            let keep = pool.len() / 2;
            let remove = pool.len() - keep;
            for _ in 0..remove {
                pool.pop_back();
                removed += 1;
            }
        }
        
        inner.stats.available_buffers = inner.stats.available_buffers.saturating_sub(removed);
        Ok(())
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BufferPool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// Thread-safe global buffer pools for common sizes
static mut GLOBAL_POOLS: Option<GlobalPools> = None;
static INIT_GLOBAL: std::sync::Once = std::sync::Once::new();

struct GlobalPools {
    small_pool: BufferPool,    // 1KB buffers
    medium_pool: BufferPool,   // 64KB buffers  
    large_pool: BufferPool,    // 1MB buffers
}

fn init_global_pools() {
    unsafe {
        GLOBAL_POOLS = Some(GlobalPools {
            small_pool: BufferPool::with_max_buffers(128),
            medium_pool: BufferPool::with_max_buffers(64),
            large_pool: BufferPool::with_max_buffers(32),
        });
    }
}

fn get_global_pools() -> &'static GlobalPools {
    INIT_GLOBAL.call_once(init_global_pools);
    unsafe { GLOBAL_POOLS.as_ref().unwrap() }
}

/// Get a small buffer (1KB) from the global pool
pub fn get_small_buffer() -> Result<PooledBuffer, BufferPoolError> {
    get_global_pools().small_pool.get_buffer(1024)
}

/// Get a medium buffer (64KB) from the global pool  
pub fn get_medium_buffer() -> Result<PooledBuffer, BufferPoolError> {
    get_global_pools().medium_pool.get_buffer(64 * 1024)
}

/// Get a large buffer (1MB) from the global pool
pub fn get_large_buffer() -> Result<PooledBuffer, BufferPoolError> {
    get_global_pools().large_pool.get_buffer(1024 * 1024)
}

/// Get a custom-sized buffer from the appropriate global pool
pub fn get_buffer(size: usize) -> Result<PooledBuffer, BufferPoolError> {
    let pools = get_global_pools();
    
    if size <= 1024 {
        pools.small_pool.get_buffer(size)
    } else if size <= 64 * 1024 {
        pools.medium_pool.get_buffer(size)
    } else {
        pools.large_pool.get_buffer(size)
    }
}

/// Get statistics for all global pools
pub fn get_global_stats() -> Result<(BufferPoolStats, BufferPoolStats, BufferPoolStats), BufferPoolError> {
    let pools = get_global_pools();
    Ok((
        pools.small_pool.stats()?,
        pools.medium_pool.stats()?,
        pools.large_pool.stats()?,
    ))
}

/// Clear all global buffer pools
pub fn clear_global_pools() -> Result<(), BufferPoolError> {
    let pools = get_global_pools();
    pools.small_pool.clear()?;
    pools.medium_pool.clear()?;
    pools.large_pool.clear()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_pool_basic() {
        let pool = BufferPool::new();
        let mut buffer = pool.get_buffer(1024).unwrap();
        
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.len(), 1024);
        
        // Write some data
        buffer.as_mut_slice()[0] = 42;
        assert_eq!(buffer.as_slice()[0], 42);
    }
    
    #[test]
    fn test_buffer_reuse() {
        let pool = BufferPool::new();
        
        // Get a buffer and drop it
        {
            let _buffer = pool.get_buffer(1024).unwrap();
        }
        
        // Get another buffer - should reuse the previous one
        let stats_before = pool.stats().unwrap();
        let _buffer2 = pool.get_buffer(1024).unwrap();
        let stats_after = pool.stats().unwrap();
        
        assert_eq!(stats_after.total_allocated, stats_before.total_allocated);
    }
    
    #[test]
    fn test_global_pools() {
        let mut small = get_small_buffer().unwrap();
        let mut medium = get_medium_buffer().unwrap();
        let mut large = get_large_buffer().unwrap();
        
        assert!(small.capacity() >= 1024);
        assert!(medium.capacity() >= 64 * 1024);
        assert!(large.capacity() >= 1024 * 1024);
        
        // Test writing to each
        small.as_mut_slice()[0] = 1;
        medium.as_mut_slice()[0] = 2;  
        large.as_mut_slice()[0] = 3;
        
        assert_eq!(small.as_slice()[0], 1);
        assert_eq!(medium.as_slice()[0], 2);
        assert_eq!(large.as_slice()[0], 3);
    }
}