FRD-PU: The Fast RAM Data-Processing Unit
A high-performance, zero-dependency library built from the ground up for extreme efficiency. It is designed to handle massive computational tasks and data streams with minimal resource consumption. This library is ideal for creating hyper-fast applications without a monstrous hardware footprint.

Our philosophy is simple: Do more with less. We achieve this through a unique blend of mathematical algorithms and zero-copy data streaming, all built on a truly dependency-free foundation. This gives you the power to create professional, dominant applications that make bloated, resource-hogging software a thing of the past.

Core Features
Absolute 0 Dependencies: We rely only on the Rust standard library, ensuring a tiny footprint and lightning-fast compilation.

Memory-First Design: The library's core is built to avoid unnecessary memory allocations, allowing you to process massive datasets with minimal memory impact.

Optimized Engines: We provide specialized APIs for different types of computation.

The Full Guide to the Library
1. cpu_task: Sequential Task Execution
The cpu_task module provides a simple, ergonomic API for wrapping and executing a single, sequential task. It is useful for encapsulating a unit of work that needs to be performed on a single thread. This design is highly efficient for tasks that do not benefit from parallelization.

new_cpu_task<I, O, T>(input: I, task: T)
Signature: pub fn new_cpu_task<I, O, T>(input: I, task: T) -> CpuTask<I, O, T> where T: FnOnce(I) -> O,

Purpose: This function serves as a constructor, creating a new CpuTask instance. It encapsulates the data to be processed (input) and the work to be done (task) into a single, cohesive unit. The FnOnce trait bound ensures that the task closure can be executed exactly once, which is a common and efficient pattern for one-off operations.

Arguments:

input: The data that the task will operate on.

task: A closure that defines the work to be performed on the input data.

Returns: A CpuTask struct, ready to be executed.

CpuTask::execute()
Signature: pub fn execute(&mut self) -> Result<O, CpuTaskError>

Purpose: This method is the entry point for running the defined task. It consumes the stored input data, executes the closure, and returns the result of the computation. The use of Result provides robust error handling, specifically for cases where the input data might be missing after a previous execution.

Returns: A Result that is Ok with the output O on success, or an Err with a CpuTaskError if the task encounters an issue.

// Example: Processing a Single Value
use frd_pu::{new_cpu_task, CpuTaskError};

fn main() -> Result<(), CpuTaskError> {
    // Define a task that calculates the square of a number.
    let input = 123456789;
    let task = new_cpu_task(input, |x| x * x);
    
    // Execute the task and handle the result.
    let result = task.execute()?;
    println!("The square of {} is {}", input, result);
    
    Ok(())
}

2. parallel: Data-Parallel Execution
The parallel module is the powerhouse for data-parallel tasks. It efficiently distributes a large workload across multiple CPU cores to dramatically speed up computation. This module is the go-to for tasks that can be broken down into independent sub-tasks, such as processing a large vector of data. It utilizes std::thread::scope for safe, zero-dependency concurrency.

execute_parallel<I, O, F>(input: Vec<I>, workers: usize, task: F)
Signature: pub fn execute_parallel<I, O, F>(mut input: Vec<I>, workers: usize, task: F) -> Result<Vec<O>, ParallelTaskError> where I: Send + 'static, O: Send + 'static, F: Fn(&I) -> O + Send + Sync + UnwindSafe + 'static,

Purpose: This function is the core of the parallel processing engine. It takes a vector of data, splits it into chunks, and spawns a thread to process each chunk concurrently. This approach maximizes CPU utilization and is ideal for "embarrassingly parallel" tasks. It uses std::thread::scope to ensure that all child threads are joined before the function returns, preventing potential data races and memory issues.

Arguments:

input: The vector of data to be processed. The data type I must be Send and 'static to be safely moved between threads.

workers: The number of worker threads to spawn. If 0, it will automatically use the number of available system cores for optimal performance.

task: A closure that defines the work to be done on each element. It must be Send and Sync to be safely shared across multiple threads. The UnwindSafe trait ensures that a panic in one thread does not cause undefined behavior.

Returns: A Result that is Ok with the vector of processed data O on success, or an Err with a ParallelTaskError if a worker thread panics.

// Example: Parallel Vector Processing
use frd_pu::{execute_parallel, ParallelTaskError};

fn main() -> Result<(), ParallelTaskError> {
    // Create a large vector of numbers.
    let input: Vec<i32> = (0..1_000_000).collect();
    
    // Define a task to double each number.
    let double_task = |x: &i32| x * 2;
    
    // Execute the task in parallel using all available cores (workers = 0).
    let doubled_values = execute_parallel(input, 0, double_task)?;
    
    // Print a few results to verify.
    println!("First 5 doubled values: {:?}", &doubled_values[0..5]);
    
    Ok(())
}

3. data_stream: Memory-Efficient I/O
The data_stream module provides a lean and mean way to process large files without loading the entire thing into memory. It reads files in manageable chunks, making it perfect for handling multi-gigabyte or even multi-terabyte files without crashing your app.

new_file_stream<P: AsRef<Path>>(path: P, chunk_size: usize)
Signature: pub fn new_file_stream<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<DataStream<File>, DataStreamError>

Purpose: This function constructs a DataStream specifically for a file. It opens the file at the given path and prepares the internal buffer for chunked reading. This is the primary entry point for processing local file data.

Arguments:

path: The file path to the data source. It accepts anything that can be converted to a Path, providing flexibility.

chunk_size: The size of the buffer for each read operation. A carefully chosen chunk size can optimize performance based on the system's I/O characteristics.

Returns: A Result that is Ok with a DataStream instance on success, or an Err with a DataStreamError if the file cannot be opened or if the chunk size is invalid.

new_network_stream(stream: TcpStream, chunk_size: usize)
Signature: pub fn new_network_stream(stream: TcpStream, chunk_size: usize) -> Result<DataStream<TcpStream>, DataStreamError>

Purpose: This function is similar to new_file_stream but is tailored for network sockets. It constructs a DataStream over an existing TcpStream, allowing you to process incoming network data in chunks without holding large amounts of data in memory.

Arguments:

stream: An established TcpStream from which to read data.

chunk_size: The size of the buffer for each network read.

Returns: A Result that is Ok with a DataStream instance on success, or an Err with a DataStreamError if the chunk size is invalid.

DataStream::process_chunks<F, E>(&mut self, processor: F)
Signature: pub fn process_chunks<F, E>(&mut self, mut processor: F) -> Result<(), DataStreamError> where F: FnMut(&[u8]) -> Result<(), E>, E: Into<String>,

Purpose: This is the workhorse of the DataStream. It enters a loop, continuously reading data from the source in chunks defined by the chunk_size until the end of the stream is reached. It then applies a user-provided processor closure to each chunk. This design enables a "streaming" pipeline where data is processed as it arrives, conserving memory.

Arguments:

processor: A mutable closure that takes a byte slice (&[u8]) of the current chunk and performs the desired work. It returns a Result to allow the caller to signal an error and stop the stream processing.

Returns: A Result that is Ok(()) if the entire stream is processed without error, or an Err with a DataStreamError if an I/O error or processor error occurs.

// Example: Streaming a File
use frd_pu::data_stream::{new_file_stream, DataStreamError};
use std::io::{Result as IoResult, Write, Read};
use std::fs::File;

fn main() -> Result<(), DataStreamError> {
    // First, create a dummy file to read from.
    let file_path = "large_file.txt";
    let mut file = File::create(file_path).map_err(DataStreamError::from)?;
    file.write_all(b"Hello, world! This is a test file for our data stream library.").map_err(DataStreamError::from)?;

    // Create a new FileStream instance.
    let mut stream = new_file_stream(file_path, 10)?;
    
    // Process the file in chunks and print each chunk.
    let mut chunk_count = 0;
    stream.process_chunks(|chunk| {
        chunk_count += 1;
        println!("Chunk {}: {}", chunk_count, String::from_utf8_lossy(chunk));
        Ok(())
    })?;

    // Clean up the dummy file.
    std::fs::remove_file(file_path).map_err(DataStreamError::from)?;
    
    Ok(())
}

4. bloom_filter: Probabilistic Set Membership
The bloom_filter module implements a memory-efficient, probabilistic data structure. It is used to test for the existence of an element in a large set, with a small chance of false positives. It is perfect for scenarios where memory usage is critical, such as checking for the existence of an item in a massive dataset.

BloomFilter::new(capacity: usize, false_positive_probability: f64)
Signature: pub fn new(capacity: usize, false_positive_probability: f64) -> Result<Self, BloomFilterError>

Purpose: This is the constructor for the BloomFilter. It calculates the optimal size of the bit vector (m) and the number of hash functions (k) required to achieve the desired false positive probability for a given capacity. This ensures the best possible performance-to-memory trade-off.

Arguments:

capacity: The expected number of items to be stored in the filter.

false_positive_probability: The acceptable probability of a false positive, expressed as a floating-point number between 0.0 and 1.0 (exclusive).

Returns: A Result that is Ok with a new BloomFilter on success, or an Err with a BloomFilterError if the arguments are invalid.

BloomFilter::add<T: Hash>(&mut self, item: &T)
Signature: pub fn add<T: Hash + ?Sized>(&mut self, item: &T)

Purpose: This method adds an item to the filter. It computes k different hash values for the item and sets the corresponding bits in the internal bit vector. The ?Sized trait bound allows for flexibility with item types, such as string slices.

Arguments:

item: The item to be added to the filter. The type T must implement the Hash trait.

BloomFilter::check<T: Hash>(&self, item: &T)
Signature: pub fn check<T: Hash + ?Sized>(&self, item: &T) -> bool

Purpose: This method checks if an item may be in the set. It computes the same k hash values as the add method and checks if all corresponding bits are set in the bit vector. If even one bit is not set, the item is definitively not in the set. If all are set, it's highly probable that the item is present.

Arguments:

item: The item to check for existence.

Returns: true if the item is probably in the set, and false if it is definitely not in the set.

// Example: Using a Bloom Filter
use frd_pu::{new_bloom_filter, BloomFilterError};

fn main() -> Result<(), BloomFilterError> {
    // Create a new Bloom filter with a capacity of 1000 items
    // and a false positive probability of 0.01.
    let mut filter = new_bloom_filter(1000, 0.01)?;
    
    // Add some items to the filter.
    filter.add(&"rust");
    filter.add(&"programming");
    filter.add(&"efficiency");
    
    // Check for existing and non-existing items.
    println!("Does 'rust' exist? {}", filter.check(&"rust")); // Expected: true
    println!("Does 'python' exist? {}", filter.check(&"python")); // Expected: false
    
    Ok(())
}

5. cache: High-Performance LRU Cache
This module contains a Least Recently Used (LRU) cache that is memory-aware and designed for speed and efficiency. It uses a combination of a hash map for fast lookups and a linked list for efficient access-order tracking. The cache is managed by both a maximum item count and a total size in bytes.

LruCache::new(max_items: usize, max_size: usize)
Signature: pub fn new(max_items: usize, max_size: usize) -> LruCache<K, V>

Purpose: This constructor initializes a new, empty LruCache. It sets the capacity limits for both the number of items and the total memory usage.

Arguments:

max_items: The maximum number of key-value pairs the cache can hold.

max_size: The maximum total memory size in bytes that the cache can occupy.

Returns: A new LruCache instance.

LruCache::insert(key: K, value: V, size: usize)
Signature: pub fn insert(&mut self, key: K, value: V, size: usize) -> Result<(), CacheError>

Purpose: Inserts a new key-value pair into the cache. The size argument allows the user to specify the memory footprint of the value, giving precise control over memory management. If the cache is full (either by item count or total size), it will automatically evict the least recently used item to make space.

Arguments:

key: The key for the item.

value: The value to be stored.

size: The size of the value in bytes.

Returns: A Result that is Ok(()) on success, or an Err with a CacheError if the item to be inserted is larger than the cache's maximum size.

LruCache::get(&self, key: &K)
Signature: pub fn get(&mut self, key: &K) -> Option<&V>

Purpose: Retrieves a value from the cache using its key. A successful retrieval marks the item as the "most recently used" by moving it to the front of the internal LRU list. This is a key part of the LRU eviction policy.

Arguments:

key: A reference to the key of the item to retrieve.

Returns: An Option containing a reference to the value if the key is found, or None if the key is not in the cache.

LruCache::remove(&mut self, key: &K)
Signature: pub fn remove(&mut self, key: &K) -> Option<V>

Purpose: Removes a key-value pair from the cache and returns the value. This function also updates the internal item count and size to reflect the removal.

Arguments:

key: A reference to the key of the item to remove.

Returns: An Option containing the removed value if the key was found, or None if the key was not in the cache.

// Example: Using the LRU Cache
use frd_pu::cache::LruCache;

fn main() {
    let mut cache = LruCache::new(2, 100);
    
    // Insert some items.
    cache.insert("key1".to_string(), "val1".to_string(), 4).unwrap();
    cache.insert("key2".to_string(), "val2".to_string(), 4).unwrap();
    
    // Access "key1", making it the most recently used.
    let val1 = cache.get(&"key1".to_string());
    println!("Found value for key1: {:?}", val1);

    // Insert a new item, which should evict "key2" because it is now the least recently used.
    cache.insert("key3".to_string(), "val3".to_string(), 4).unwrap();
    
    // Check if "key2" was evicted.
    let val2 = cache.get(&"key2".to_string());
    println!("Found value for key2: {:?}", val2); // Should be None
}

6. concurrent: Thread-Safe Data Structures
This module provides thread-safe wrappers for common data structures, allowing for safe concurrent access from multiple threads. It uses std::sync::Mutex and std::sync::Arc to provide exclusive access and shared ownership, respectively.

ConcurrentList::new()
Signature: pub fn new() -> ConcurrentList<T>

Purpose: This constructor creates a new, empty, thread-safe list. It internally initializes a Vec wrapped in a Mutex to manage concurrent access.

Returns: A new ConcurrentList instance.

ConcurrentList::push(item: T)
Signature: pub fn push(&self, item: T) -> Result<(), ConcurrentListError>

Purpose: Appends an item to the end of the list in a thread-safe manner. It acquires a lock on the internal Mutex, pushes the item, and then releases the lock.

Arguments:

item: The item to be pushed.

Returns: A Result that is Ok(()) on success, or an Err with a ConcurrentListError if the mutex lock cannot be acquired.

ConcurrentList::pop()
Signature: pub fn pop(&self) -> Result<Option<T>, ConcurrentListError>

Purpose: Removes and returns the last item from the list in a thread-safe manner. This is a common operation in multi-producer, multi-consumer scenarios.

Returns: A Result that is Ok with an Option containing the removed item if the list is not empty, or an Err with a ConcurrentListError if the mutex lock cannot be acquired.

ConcurrentList::len()
Signature: pub fn len(&self) -> Result<usize, ConcurrentListError>

Purpose: Returns the number of items currently in the list in a thread-safe manner.

Returns: A Result that is Ok with the list's length on success, or an Err with a ConcurrentListError if the mutex lock cannot be acquired.

// Example: Using a Concurrent List
use frd_pu::concurrent::ConcurrentList;
use std::thread;
use std::sync::Arc;

fn main() {
    // Create a new thread-safe list.
    let list = Arc::new(ConcurrentList::new());
    let mut handles = vec![];

    // Spawn 10 threads to push items to the list concurrently.
    for i in 0..10 {
        let list_clone = Arc::clone(&list);
        let handle = thread::spawn(move || {
            list_clone.push(i).unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all threads to finish.
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Check the final size of the list.
    println!("Final list length: {}", list.len().unwrap());
}

7. hasher: High-Performance Hashing Engine
This module provides a zero-dependency, high-performance hashing engine. It is ideal for general-purpose hashing where collision resistance is not a primary concern. The hash_stream function is particularly useful for hashing large files or data streams without consuming excessive memory.

hash_bytes(data: &[u8])
Signature: pub fn hash_bytes(data: &[u8]) -> u64

Purpose: Computes a 64-bit hash from a byte slice. This is a fast, in-memory operation suitable for small data inputs.

Arguments:

data: The byte slice to be hashed.

Returns: The resulting 64-bit hash as a u64.

hash_file<P: AsRef<Path>>(path: P)
Signature: pub fn hash_file<P: AsRef<Path>>(path: P) -> Result<u64, HasherError>

Purpose: This function provides a convenient way to hash the contents of a file. It opens the file and passes the file stream to the more general hash_stream function, ensuring memory efficiency for large files.

Arguments:

path: The path to the file.

Returns: A Result that is Ok with the 64-bit hash on success, or an Err with a HasherError if a file I/O error occurs.

hash_stream<R: Read>(reader: R)
Signature: pub fn hash_stream<R: Read>(mut reader: R) -> Result<u64, HasherError>

Purpose: This is the core, generic hashing function. It allows for hashing data from any source that implements the Read trait (e.g., a file, a network stream, or an in-memory buffer). It reads the data in small chunks to ensure low memory usage, making it extremely efficient for large data sets.

Arguments:

reader: A type that implements the Read trait, serving as the data source.

Returns: A Result that is Ok with the 64-bit hash on success, or an Err with a HasherError if an I/O error occurs.

// Example: Hashing a File Stream
use frd_pu::hasher::hash_stream;
use std::io::Cursor;

fn main() {
    // Use `Cursor` to simulate a stream from an in-memory byte slice.
    let data = b"FRD-PU is the GOAT.";
    let reader = Cursor::new(data);

    let result = hash_stream(reader);
    println!("Hash of data: {:?}", result.unwrap());
}

8. btree: Binary Search Tree
The btree module provides a professional, zero-dependency implementation of a Binary Search Tree. This data structure is ideal for storing key-value pairs in a sorted order, allowing for efficient search, insertion, and deletion operations.

BTree::new()
Signature: pub fn new() -> BTree<K, V>

Purpose: This constructor creates a new, empty Binary Search Tree.

Returns: A new BTree instance.

BTree::insert(key: K, value: V)
Signature: pub fn insert(&mut self, key: K, value: V) -> Result<(), BTreeError>

Purpose: Inserts a new key-value pair into the tree. It traverses the tree to find the correct position for the new node, maintaining the sorted property.

Arguments:

key: The key for the item. The type K must implement the Ord trait for ordering.

value: The value to be stored.

Returns: A Result that is Ok(()) on success, or an Err with a BTreeError if the key already exists.

BTree::search(&self, key: &K)
Signature: pub fn search(&self, key: &K) -> Option<&V>

Purpose: Searches for a key in the tree. It efficiently traverses the tree, comparing the target key at each node to quickly find the value.

Arguments:

key: A reference to the key to search for.

Returns: An Option containing a reference to the value if the key is found, or None if the key is not in the tree.

BTree::in_order_traversal()
Signature: pub fn in_order_traversal(&self) -> Vec<(&K, &V)>

Purpose: Performs an in-order traversal of the tree, collecting all key-value pairs into a vector. Because of the tree's structure, the resulting vector will be sorted by key.

Returns: A Vec containing references to the key-value pairs in sorted order.

// Example: Using a Binary Search Tree
use frd_pu::btree::BTree;

fn main() {
    let mut btree = BTree::new();

    // Insert key-value pairs.
    btree.insert(5, "apple").unwrap();
    btree.insert(3, "banana").unwrap();
    btree.insert(8, "cherry").unwrap();

    // Search for a value.
    let result = btree.search(&3);
    println!("Found value for key 3: {:?}", result);

    // Perform an in-order traversal.
    let sorted_items = btree.in_order_traversal();
    println!("Sorted items: {:?}", sorted_items);
}

9. trie: Prefix Tree
The trie module implements a zero-dependency Trie (Prefix Tree) data structure. This is a tree-like data structure used to store a dynamic set or associative array where the keys are usually strings. Tries are highly efficient for tasks like autocompletion, spell-checking, and finding all keys with a common prefix.

Trie::new()
Signature: pub fn new() -> Trie

Purpose: This constructor creates a new, empty Trie with a single root node.

Returns: A new Trie instance.

Trie::insert(word: &str)
Signature: pub fn insert(&mut self, word: &str) -> Result<(), TrieError>

Purpose: Inserts a word into the Trie. It traverses the tree, creating new nodes for each character in the word that does not yet exist. The final node is marked as the end of a complete word.

Arguments:

word: The string slice to be inserted.

Returns: A Result that is Ok(()) on success, or an Err with a TrieError if an invalid character is encountered.

Trie::search(word: &str)
Signature: pub fn search(&self, word: &str) -> bool

Purpose: Checks if a complete word exists in the Trie. It traverses the tree along the path defined by the word's characters and returns true only if the final node exists and is marked as an end-of-word node.

Arguments:

word: The string slice to search for.

Returns: true if the word is found, otherwise false.

Trie::starts_with(prefix: &str)
Signature: pub fn starts_with(&self, prefix: &str) -> bool

Purpose: Checks if a prefix exists in the Trie. It traverses the tree along the path of the prefix characters. Unlike search, it only needs to confirm that a node exists for the final character of the prefix, regardless of whether it marks the end of a word.

Arguments:

prefix: The string slice to search for.

Returns: true if the prefix is found, otherwise false.

// Example: Using the Trie
use frd_pu::trie::Trie;

fn main() {
    let mut trie = Trie::new();
    
    // Insert some words.
    trie.insert("rust").unwrap();
    trie.insert("rocket").unwrap();
    trie.insert("run").unwrap();
    
    // Check if words and prefixes exist.
    println!("Does 'rust' exist? {}", trie.search("rust")); // true
    println!("Does 'run' exist? {}", trie.search("run")); // true
    println!("Does 'ru' exist as a prefix? {}", trie.starts_with("ru")); // true
    println!("Does 'go' exist? {}", trie.search("go")); // false
}

10. quicksort: In-Place Sorting Algorithm
The quicksort module provides an insanely fast in-place sorting algorithm. It is generic over any type T that implements PartialOrd for comparison, making it highly flexible. This zero-dependency implementation uses the Lomuto partition scheme and recursion to sort data efficiently.

quicksort<T: PartialOrd>(slice: &mut [T])
Signature: pub fn quicksort<T: PartialOrd>(slice: &mut [T]) -> Result<(), QuickSortError>

Purpose: This is the public-facing function that initiates the QuickSort algorithm. It takes a mutable slice and sorts its elements in place. The function handles the initial setup and calls the internal recursive function to perform the sorting.

Arguments:

slice: The mutable slice of data to be sorted.

Returns: A Result that is Ok(()) on success, or an Err with a QuickSortError if the provided slice is empty.

// Example: Sorting a Vector
use frd_pu::quicksort::quicksort;

fn main() {
    let mut data = vec![5, 2, 8, 1, 9];
    quicksort(&mut data).unwrap();
    
    println!("Sorted data: {:?}", data); // [1, 2, 5, 8, 9]
}
