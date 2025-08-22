// src/data_stream.rs

use std::error::Error;
use std::fs::File;
use std::io::{self, Read, Result as IoResult};
use std::path::Path;
use std::fmt;

/// Error type for file streaming operations.
#[derive(Debug)]
pub enum FileStreamError {
    /// Indicates an I/O error occurred during a file stream operation.
    IoError(io::Error),
    /// Indicates that the provided chunk size was invalid.
    InvalidChunkSize,
}

impl From<io::Error> for FileStreamError {
    fn from(error: io::Error) -> Self {
        FileStreamError::IoError(error)
    }
}

impl Error for FileStreamError {}

impl fmt::Display for FileStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileStreamError::IoError(e) => write!(f, "I/O error during file stream: {}", e),
            FileStreamError::InvalidChunkSize => {
                write!(f, "Invalid chunk size: must be greater than 0.")
            }
        }
    }
}

/// A struct for reading and processing a file in chunks.
pub struct FileStream {
    file: File,
    buffer: Vec<u8>,
}

impl FileStream {
    /// Creates a new `FileStream` with a specified chunk size.
    ///
    /// # Arguments
    /// * `path` - The path to the file to be streamed.
    /// * `chunk_size` - The size of each data chunk to be read.
    ///
    /// # Returns
    /// A `Result` containing the new `FileStream` instance, or an error if
    /// the chunk size is invalid or the file cannot be opened.
    pub fn new<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<Self, FileStreamError> {
        if chunk_size == 0 {
            return Err(FileStreamError::InvalidChunkSize);
        }
        let file = File::open(path)?;
        Ok(FileStream {
            file,
            buffer: vec![0; chunk_size],
        })
    }

    /// Processes the file chunk by chunk with a provided closure.
    ///
    /// This function reads the file in chunks and applies a given processor
    /// function to each chunk.
    ///
    /// # Arguments
    /// * `processor` - A mutable closure that processes each chunk. It takes
    /// a byte slice of the chunk and returns an `IoResult`.
    ///
    /// # Returns
    /// An `IoResult` indicating success or failure of the streaming operation.
    pub fn process_chunks<F>(&mut self, mut processor: F) -> IoResult<()>
    where
        F: FnMut(&[u8]) -> IoResult<()>,
    {
        loop {
            let bytes_read = self.file.read(&mut self.buffer)?;
            if bytes_read == 0 {
                break; // End of file
            }
            processor(&self.buffer[..bytes_read])?;
        }
        Ok(())
    }
}
