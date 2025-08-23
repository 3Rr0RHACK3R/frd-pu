// src/parallel.rs

use std::error::Error;
use std::fmt;
use std::panic::{self, UnwindSafe};
use std::thread;
use std::marker::Send;

/// Error type for parallel task execution.
#[derive(Debug, PartialEq)]
pub enum ParallelTaskError {
    /// Indicates that a worker thread panicked during execution.
    ThreadPanic,
}

impl fmt::Display for ParallelTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParallelTaskError::ThreadPanic => write!(f, "A worker thread panicked during execution."),
        }
    }
}

impl Error for ParallelTaskError {}

/// Executes a data-parallel task across multiple CPU cores.
///
/// This function chunks the input data and processes each chunk on a separate thread.
/// It ensures thread safety and handles panics gracefully.
///
/// # Arguments
/// * `input` - The input vector of data to process.
/// * `workers` - The number of worker threads. If 0, it defaults to the number of available system cores.
/// * `task` - The closure to apply to each element.
///
/// # Returns
/// A `Result` containing the vector of processed data in the original order, or an error if a thread panics.
pub fn execute_parallel<I, O, F>(
    input: Vec<I>,
    workers: usize,
    task: F,
) -> Result<Vec<O>, ParallelTaskError>
where
    I: Send + Sync + UnwindSafe,
    O: Send,
    F: Fn(&I) -> O + Send + Sync + UnwindSafe + 'static,
{
    // Determine the number of workers.
    let num_workers = if workers == 0 {
        thread::available_parallelism().map_or(1, |p| p.get())
    } else {
        workers
    };

    // Calculate the chunk size.
    let chunk_size = (input.len() + num_workers - 1) / num_workers;

    // Use `catch_unwind` to gracefully handle panics from worker threads.
    let result = panic::catch_unwind(move || {
        thread::scope(|s| {
            // Split the input into chunks and process each on a new thread.
            // We use `Vec::drain` to move the chunks out of the main vector,
            // which is more efficient.
            let mut handles = Vec::with_capacity(num_workers);
            let chunks_iter = input.chunks(chunk_size);

            for chunk in chunks_iter {
                let task_ref = &task;
                let handle = s.spawn(move || {
                    let mut results_for_chunk = Vec::with_capacity(chunk.len());
                    for item in chunk {
                        results_for_chunk.push(task_ref(item));
                    }
                    results_for_chunk
                });
                handles.push(handle);
            }
            
            // Collect the results from each thread, preserving the original order.
            let mut final_results = Vec::with_capacity(input.len());
            for handle in handles {
                // `handle.join()` will panic if the thread panicked.
                final_results.extend(handle.join().unwrap());
            }

            final_results
        })
    });

    match result {
        Ok(results) => Ok(results),
        Err(_) => Err(ParallelTaskError::ThreadPanic),
    }
}