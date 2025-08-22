// src/data_stream.rs

use std::error::Error;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::fmt;

/// Error type for file streaming operations.
#[derive(Debug)]
pub enum FileStreamError {
    IoError(io::Error),
    InvalidChunkSize,
}

impl From<io::Error> for FileStreamError {
    fn from(error: io.Error) -> Self {
        FileStreamError::IoError(error)
    }
}

impl Error for FileStreamError {}

impl fmt::Display for FileStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileStreamError::IoError(e) => write!(f, "I/O error during file stream: {}", e),
            FileStreamError::InvalidChunkSize => write!(f, "Invalid chunk size: must be greater than 0."),
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
    pub fn process_chunks<F>(&mut self, mut processor: F) -> Result<(), io::Error>
    where
        F: FnMut(&[u8]) -> Result<(), io::Error>,
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

/// A convenience function to create a new `FileStream`.
pub fn new_file_stream<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<FileStream, FileStreamError> {
    FileStream::new(path, chunk_size)
}
