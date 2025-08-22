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

// Public Modules
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

// Re-export the public APIs for easy access.
pub use bloom_filter::{BloomFilter, BloomFilterError};
pub use cpu_task::{CpuTask, CpuTaskError, new_cpu_task};
pub use data_stream::{DataStream, DataStreamError, new_file_stream, new_network_stream};
pub use parallel::{execute_parallel, ParallelTaskError};
pub use cache::{LruCache, CacheError};
pub use concurrent::{ConcurrentList, ConcurrentListError};
pub use hasher::{hash_bytes, hash_file, hash_stream, HasherError};
pub use btree::{BTree, BTreeError};
pub use trie::{Trie, TrieError};
pub use quicksort::{quicksort, QuickSortError};