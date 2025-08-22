// src/parallel.rs

use std::error::Error;
use std::fmt;
use std::thread;
use std::panic::{self, UnwindSafe};
use std::marker::PhantomData; // Required for generic type UnwindSafe bound

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
    // The closure F must be callable with a reference, be safe to send between threads (Send),
    // be safe to share between threads (Sync), and be safe to unwind across a panic boundary (UnwindSafe).
    F: Fn(&I) -> O + Send + Sync + UnwindSafe,
    // The input items I must be safe to send between threads and be safely shareable via
    // immutable references between threads (Sync). They also need to be unwind-safe.
    I: Send + Sync + UnwindSafe,
    // The output items O must be safe to send between threads and be unwind-safe.
    O: Send + UnwindSafe,
{
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Determine the number of threads to use.
    let num_workers = if workers > 0 {
        workers
    } else {
        thread::available_parallelism().map_or(1, |n| n.get())
    };

    let mut results: Vec<O> = Vec::with_capacity(input.len());
    // This is safe because we are initializing memory that will be immediately
    // written to by threads that we guarantee will complete before we return.
    #[allow(unsafe_code)]
    unsafe {
        results.set_len(input.len());
    }

    // The `PhantomData` fields are used to satisfy the `UnwindSafe`
    // trait bounds for `I` and `O`. Since `Vec<I>` and `Vec<O>` are
    // moved into the closure, they must be `UnwindSafe`.
    let input_phantom = PhantomData::<I>;
    let results_phantom = PhantomData::<O>;

    // Use catch_unwind to handle panics in worker threads.
    let result = panic::catch_unwind(move || {
        thread::scope(|s| {
            let chunk_size = (input.len() + num_workers - 1) / num_workers;
            if chunk_size == 0 { return; }

            // Split the input and results vectors into chunks for parallel processing.
            // These chunks are then moved into the worker threads.
            let input_chunks = input.chunks(chunk_size);
            let mut results_chunks = results.chunks_mut(chunk_size);

            for (input_chunk, results_chunk) in input_chunks.zip(&mut results_chunks) {
                let task_ref = &task;
                s.spawn(move || {
                    for (input_item, output_item) in input_chunk.iter().zip(results_chunk.iter_mut()) {
                        *output_item = task_ref(input_item);
                    }
                });
            }
        });

        results
    });

    result.map_err(|_| ParallelTaskError::ThreadPanic)
}
