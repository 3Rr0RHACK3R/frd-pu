FRD-PU: The Fast RAM Data-Processing Unit
A high-performance, zero-dependency library built from the ground up for extreme efficiency. It is designed to handle massive computational tasks and data streams with minimal resource consumption. This library is ideal for creating hyper-fast applications without a monstrous hardware footprint.

Our philosophy is simple: Do more with less. We achieve this through a unique blend of mathematical algorithms and zero-copy data streaming, all built on a truly dependency-free foundation. This gives you the power to create professional, dominant applications that make bloated, resource-hogging software a thing of the past.

Core Features
Absolute 0 Dependencies: We rely only on the Rust standard library, ensuring a tiny footprint and lightning-fast compilation.

Memory-First Design: The library's core is built to avoid unnecessary memory allocations, allowing you to process massive datasets with minimal memory impact.

Optimized Engines: We provide specialized APIs for different types of computation:

cpu_task: For single-threaded, sequential tasks.

parallel: For data-parallel tasks that leverage multiple CPU cores.

data_stream: For handling large files efficiently with zero-copy streaming.

bloom_filter: For memory-efficient, probabilistic set checks.

Getting Started
To use this library in your project, simply add it as a dependency in your Cargo.toml file:

[dependencies]
frd_pu = "0.1.0"

The Full Guide to the Library
1. The cpu_task Module
The cpu_task module provides a simple, ergonomic API for wrapping and executing a single, sequential task. It is useful for encapsulating a unit of work that needs to be performed on the main thread.

API Reference
new_cpu_task<I, O, T>(input: I, task: T) -> CpuTask<I, O, T>: Creates a new CPU task with the given input and a closure to be executed.

CpuTask::execute(): Executes the task and returns a Result containing the output or an error if execution fails.

Example: Processing a Single Value
This example shows how to use CpuTask to perform a simple, long-running calculation on a single value.

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

2. The parallel Module
The parallel module is the powerhouse for data-parallel tasks. It efficiently distributes a large workload across multiple CPU cores to dramatically speed up computation. This module is the go-to for tasks that can be broken down into independent sub-tasks, such as processing a large vector of data.

API Reference
execute_parallel<I, O, F>(input: Vec<I>, workers: usize, task: F) -> Result<Vec<O>, ParallelTaskError>: Executes a data-parallel task. The input is the vector of data to process, workers specifies the number of threads (0 for all available cores), and task is the closure to apply to each element.

Example: Parallel Vector Processing
This example demonstrates how to process a large vector in parallel.

use frd_pu::{execute_parallel, ParallelTaskError};

fn main() -> Result<(), ParallelTaskError> {
    // Create a large vector of numbers.
    let input_data: Vec<i32> = (0..1_000_000).collect();
    
    // Process the data in parallel, doubling each number.
    // Setting workers to 0 will use all available CPU cores.
    let processed_data = execute_parallel(input_data, 0, |&x| x * 2)?;

    // The result is a new vector with the processed data.
    println!("Processed the vector. First 10 elements: {:?}", &processed_data[0..10]);
    
    Ok(())
}

3. The data_stream Module
The data_stream module provides a memory-efficient way to handle very large files without loading the entire file into memory. It processes data in user-defined chunks, which is perfect for streaming data or working on files that are larger than available RAM.

API Reference
new_file_stream<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<FileStream, FileStreamError>: Creates a new file stream with a specified chunk size.

FileStream::process_chunks<F>(&mut self, mut processor: F) -> Result<(), io::Error>: Processes the file chunk-by-chunk using the provided closure.

Example: Processing a Large File
This example shows how to read a file chunk-by-chunk and print its contents.

use frd_pu::{new_file_stream, FileStreamError};
use std::io::{self, Write};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a dummy file for this example.
    File::create("example.txt")?.write_all(b"Hello from the FRD-PU data stream!")?;

    // Create a new FileStream with a chunk size of 8 bytes.
    let mut file_stream = new_file_stream("example.txt", 8)?;

    // Process each chunk and print it.
    file_stream.process_chunks(|chunk| {
        io::stdout().write_all(chunk)?;
        Ok(())
    })?;
    
    println!("\nFile processing complete.");
    
    Ok(())
}

4. The bloom_filter Module
The bloom_filter module implements a space-efficient probabilistic data structure. It is used to test whether an element is a member of a set. False positives are possible, but false negatives are not. This is perfect for scenarios where memory usage is critical, such as checking for the existence of an item in a massive dataset.

API Reference
new_bloom_filter(capacity: usize, false_positive_probability: f64) -> Result<BloomFilter, BloomFilterError>: Creates a new Bloom filter with a given capacity and acceptable false positive probability.

BloomFilter::add<T: Hash>(&mut self, item: &T): Adds an item to the filter.

BloomFilter::check<T: Hash>(&self, item: &T) -> bool: Checks if an item may be in the set.

Example: Using a Bloom Filter
This example demonstrates how to create and use a Bloom filter.

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
    println!("Does 'rust' exist? {}", filter.check(&"rust")); // true
    println!("Does 'python' exist? {}", filter.check(&"python")); // false (or true with a small chance)
    
    Ok(())
}

Contributing
We welcome contributions! Please feel free to open an issue or submit a pull request on our GitHub repository.

License
This project is licensed under the MIT License. See the LICENSE file for details.