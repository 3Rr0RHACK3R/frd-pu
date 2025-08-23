FRD-PU: The Fast RAM Data-Processing Unit
A professional-grade, high-performance, and zero-dependency library built from the ground up for extreme efficiency. This crate is designed to handle massive computational tasks and data streams with minimal resource consumption. It is ideal for creating hyper-fast applications without a monstrous hardware footprint.

Our core philosophy is simple: Do more with less. We achieve this through a unique blend of optimized algorithms and zero-copy data streaming, all built on a foundation that relies only on the Rust standard library. This empowers you to create dominant, high-performance applications that make bloated, resource-hogging software a thing of the past.

Core Features
Absolute 0 Dependencies: We rely only on the Rust standard library, ensuring a tiny footprint and lightning-fast compilation.

Memory-First Design: The library's core is built to avoid unnecessary memory allocations, allowing you to process massive datasets with minimal memory impact.

Optimized Engines: We provide specialized APIs for different types of computation, ensuring the right tool for the job.

Modules & API Documentation
This crate is composed of several powerful modules, each designed for a specific purpose.

bloom_filter
A memory-efficient, probabilistic data structure for checking if an element is a member of a set. It is ideal for scenarios where memory is a constraint and a small rate of false positives is acceptable.

BloomFilter::new(capacity, false_positive_probability): Creates a new filter with a specified expected number of items and a desired false-positive rate.

BloomFilter::insert(&self, item): Inserts a hashable item into the filter.

BloomFilter::check(&self, item): Checks if an item may be in the set. Returns false if it is definitely not, and true if it is probably in the set.

btree
A zero-dependency Binary Search Tree (BST) data structure. A BST is an efficient way to store and retrieve sorted data, providing logarithmic time complexity for search, insertion, and deletion operations on average.

BinarySearchTree::insert(&mut self, key, value): Inserts a key-value pair into the tree. Returns an error if the key already exists.

BinarySearchTree::search(&self, key): Searches for a key in the tree and returns a reference to its value, if found.

cache
A high-performance, memory-aware, Least Recently Used (LRU) cache. This cache uses a combination of a hash map for fast lookups and a linked list for efficient access-order tracking, providing O(1) average time complexity for most operations. The cache's memory usage is managed by a max_size in bytes.

LruCache::new(max_size): Creates a new cache with a maximum size in bytes.

LruCache::insert(&mut self, key, value): Inserts a key-value pair into the cache. Returns an error if the item exceeds the cache's maximum size.

LruCache::get(&mut self, key): Retrieves a value by key, marking it as the most recently used.

LruCache::remove(&mut self, key): Removes a key-value pair from the cache.

concurrent
A thread-safe list that allows for safe concurrent access from multiple threads. It wraps a standard Vec in a Mutex to provide exclusive access and uses Arc for shared ownership between threads.

ConcurrentList::new(): Creates a new, empty, thread-safe list.

ConcurrentList::push(&self, item): Appends an item to the end of the list.

ConcurrentList::pop(&self): Removes and returns the last item in the list.

ConcurrentList::get(&self, index): Returns a reference to the item at the specified index.

cpu_task
A professional wrapper for a single-threaded CPU task and its input. This module provides a clean and encapsulated way to define and execute a sequential computational task, ensuring robust error handling.

new_cpu_task(input, task): Creates a new single-threaded task with a defined input and a closure for the work to be done.

CpuTask::execute(): Executes the defined CPU task and returns the result or an error.

data_stream
An efficient API for handling large files and network streams in a chunked manner. This module abstracts over different data sources, allowing for low-memory processing of vast datasets.

new_file_stream(path, chunk_size): A convenience function to create a data stream from a file path.

new_network_stream(stream, chunk_size): A convenience function to create a data stream from a network stream.

DataStream::for_each_chunk(&mut self, processor): Reads the stream in chunks and processes each chunk with a provided closure.

hasher
A zero-dependency, high-performance hashing engine. This module provides functions for hashing byte slices, files, and data streams using a fast, non-cryptographic DefaultHasher.

hash_bytes(data): Hashes a byte slice into a 64-bit integer.

hash_file(path): Hashes the contents of a file.

hash_stream(reader): Hashes data from any type that implements the Read trait.

parallel
A powerful module for executing data-parallel tasks across multiple CPU cores. It chunks input data and processes each chunk on a separate thread, ensuring thread safety and graceful panic handling.

execute_parallel(input, workers, task): Executes a data-parallel task, distributing the work across a specified number of threads.

quicksort
An insanely fast, zero-dependency, in-place sorting algorithm. This implementation of QuickSort sorts a mutable slice in place and is generic over any type that can be compared.

quicksort(slice): Sorts a mutable slice of data in place using the QuickSort algorithm.

trie
A memory-efficient, zero-dependency Trie (Prefix Tree) data structure. A Trie is ideal for efficient retrieval of keys from a dataset of strings, making it perfect for applications like autocompletion and spell-checking.

Trie::insert(&mut self, word): Inserts a word into the Trie.

Trie::search(&self, word): Checks if a complete word exists in the Trie.

Trie::starts_with(&self, prefix): Checks if a prefix exists in the Trie.

Getting Started
To use this crate in your project, add it to your Cargo.toml.

[dependencies]
frd-pu = { path = "/path/to/your/frd-pu" }

Contributing
We welcome contributions from the community. Please read the CONTRIBUTING.md for guidelines on how to submit pull requests, report bugs, and propose new features.

License
This project is licensed under the MIT License.