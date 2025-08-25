// src/lib.rs

//! # FRD-PU: The Fast RAM Data-Processing Unit
//!
//! A high-performance, zero-dependency library built from the ground up for extreme efficiency.
//! It is designed to handle massive computational tasks and data streams with minimal resource
//! consumption. This library is ideal for creating hyper-fast applications without a monstrous
//! hardware footprint.
//!
//! Our philosophy is simple: **Do more with less.** We achieve this through a unique blend of
//! mathematical algorithms and zero-copy data streaming, all built on top of a truly
//! dependency-free foundation. This is how we give you the power to create professional,
//! dominant applications that make bloated, resource-hogging software a thing of the past.
//!
//! ## Core Features:
//!
//! * **Absolute 0 Dependencies:** We rely only on the Rust standard library, ensuring a tiny
//!     footprint and lightning-fast compilation.
//! * **Memory-First Design:** The library's core is built to avoid unnecessary memory allocations,
//!     allowing you to process massive datasets with minimal memory impact.
//! * **Optimized Engines:** We provide specialized APIs for different types of computation:
//!     * `cpu_task` for single-threaded, sequential tasks.
//!     * `parallel` for data-parallel tasks that leverage multiple CPU cores.
//!     * `data_stream` for handling large files efficiently.
//!     * `bloom_filter` for memory-efficient set checks.
//!     * `cache` for a high-performance, memory-aware Least Recently Used (LRU) cache.
//!     * `concurrent` for thread-safe data structures for safe parallel processing.
//!     * `hasher` for a zero-dependency, high-performance hashing engine.
//!     * `btree` for an efficient binary search tree.
//!     * `trie` for a memory-efficient prefix tree.
//!     * `quicksort` for an insanely fast in-place sorting algorithm.
//!     * `compression` for high-performance LZ77-style data compression and decompression.
//!     * `memory_pool` for zero-allocation memory management with pre-allocated pools.
//!     * `buffer_pool` for reusable I/O buffers that eliminate allocation/deallocation cycles.
//!     * `universal_processor` for revolutionary fractal processing that scales from bytes to terabytes.
//!     * `tcp_server` for production-ready, maximum performance TCP server with Windows optimization.

// Public Modules (ALL existing modules preserved)
pub mod bloom_filter;
pub mod cpu_task;
pub mod data_stream;
pub mod parallel;
pub mod cache;
pub mod concurrent;
pub mod hasher;
pub mod btree;
pub mod trie;
pub mod quicksort;
pub mod compression; // NEW MODULE ADDED
pub mod memory_pool; // NEW MODULE ADDED
pub mod buffer_pool; // NEW MODULE ADDED
pub mod universal_processor; // UNIVERSAL PROCESSOR MODULE ADDED
pub mod tcp_server; // TCP SERVER MODULE ADDED

// Re-export the public APIs for easy access (ALL existing re-exports preserved)
pub use bloom_filter::{BloomFilter, BloomFilterError};
pub use cpu_task::{CpuTask, CpuTaskError, new_cpu_task};
pub use data_stream::{DataStream, DataStreamError, new_file_stream, new_network_stream};
pub use parallel::{execute_parallel, ParallelTaskError};
pub use cache::{LruCache, CacheError};
pub use concurrent::{ConcurrentList, ConcurrentListError};
pub use hasher::{hash_bytes, hash_file, hash_stream, HasherError};
pub use btree::{BinarySearchTree, BinarySearchTreeError};
pub use trie::{Trie, TrieError};
pub use quicksort::{quicksort, QuickSortError};

// NEW RE-EXPORTS for compression module
pub use compression::{
    CompressionEngine, 
    CompressionError, 
    CompressionStats,
    compress_data, 
    decompress_data, 
    compress_text, 
    decompress_to_text,
    get_compression_stats
};

// NEW RE-EXPORTS for memory pool module
pub use memory_pool::{
    FixedPool,
    ObjectPool,
    PooledMemory,
    PooledObject,
    PoolManager,
    PoolStats,
    MemoryPoolError,
    create_fixed_pool,
    create_small_pool,
    create_medium_pool,
    create_large_pool
};

// NEW RE-EXPORTS for buffer pool module
pub use buffer_pool::{
    BufferPool,
    PooledBuffer,
    BufferPoolError,
    BufferPoolStats,
    get_small_buffer,
    get_medium_buffer,
    get_large_buffer,
    get_buffer,
    get_global_stats,
    clear_global_pools
};

// NEW RE-EXPORTS for universal processor module
pub use universal_processor::{
    UniversalProcessor,
    UniversalProcessorError,
    ProcessingPattern,
    ScalingBehavior,
    OptimizationMode,
    DataPattern,
    ProcessingStats,
    ProcessingContext,
    create_transform_processor,
    create_aggregate_processor,
    create_filter_processor,
    create_math_processor,
    process_adaptive,
    process_fractal
};

// NEW RE-EXPORTS for tcp_server module
pub use tcp_server::{
    TcpServer,
    TcpServerError,
    ConnectionHandler,
    ConnectionStats,
    ConnectionStatsSnapshot,
    ServerConfig,
    EchoHandler,
    HttpHandler,
    new_echo_server,
    new_http_server,
    new_tcp_server,
    new_tcp_server_with_config,
    DEFAULT_BUFFER_SIZE,
    MAX_BUFFER_SIZE,
    MIN_BUFFER_SIZE,
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_TIMEOUT_SECS,
    DEFAULT_RATE_LIMIT
};