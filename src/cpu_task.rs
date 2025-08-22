// src/cpu_task.rs

use std::error::Error;
use std::fmt;

/// Error type for single-threaded CPU task execution.
#[derive(Debug, PartialEq)]
pub enum CpuTaskError {
    ExecutionError(String),
}

impl fmt::Display for CpuTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuTaskError::ExecutionError(msg) => write!(f, "CPU task execution failed: {}", msg),
        }
    }
}

impl Error for CpuTaskError {}

/// A wrapper for a single-threaded task and its input.
pub struct CpuTask<I, O, T>
where
    T: FnOnce(I) -> O,
{
    input: Option<I>,
    task: T,
}

/// Creates a new single-threaded, sequential task.
///
/// # Arguments
/// * `input` - The data to be processed by the task.
/// * `task` - The closure that defines the work to be done.
pub fn new_cpu_task<I, O, T>(input: I, task: T) -> CpuTask<I, O, T>
where
    T: FnOnce(I) -> O,
{
    CpuTask {
        input: Some(input),
        task,
    }
}

impl<I, O, T> CpuTask<I, O, T>
where
    T: FnOnce(I) -> O,
{
    /// Executes the task, consuming the input.
    ///
    /// Returns an error if the task has already been executed.
    pub fn execute(mut self) -> Result<O, CpuTaskError> {
        let input = self.input.take().ok_or_else(|| {
            CpuTaskError::ExecutionError("Input was already consumed".to_string())
        })?;
        let output = (self.task)(input);
        Ok(output)
    }
}
