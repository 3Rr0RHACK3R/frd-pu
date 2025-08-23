// src/data_stream.rs

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::net::TcpStream;
use std::path::Path;

/// A professional-grade, zero-dependency trait for any type that can be read from in a chunked manner.
/// This abstracts over different data sources like files or network streams.
pub trait DataReader: Read {}

impl DataReader for File {}
impl DataReader for TcpStream {}

/// Error type for file and network streaming operations.
#[derive(Debug)]
pub enum DataStreamError {
    /// Indicates an I/O error occurred during a data stream operation.
    IoError(io::Error),
    /// Indicates that the provided chunk size was invalid.
    InvalidChunkSize,
    /// A generic error from the chunk processor.
    ProcessorError(String),
}

impl From<io::Error> for DataStreamError {
    fn from(error: io::Error) -> Self {
        DataStreamError::IoError(error)
    }
}

impl Error for DataStreamError {}

impl fmt::Display for DataStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataStreamError::IoError(e) => write!(f, "I/O error during data stream: {}", e),
            DataStreamError::InvalidChunkSize => {
                write!(f, "Invalid chunk size provided; must be greater than 0.")
            }
            DataStreamError::ProcessorError(e) => write!(f, "Processor function returned an error: {}", e),
        }
    }
}

/// A highly optimized data stream processor for files and network streams.
///
/// This structure is designed to be a "zero-copy" processor in the professional sense,
/// meaning it avoids any unnecessary heap allocations and data copies. It uses a
/// fixed-size buffer to read data and then hands a reference to that buffer directly
/// to a processor function. This minimizes memory overhead and maximizes speed,
/// embodying our core philosophy of "Do more with less."
pub struct DataStream<R: DataReader> {
    reader: R,
    buffer: Vec<u8>,
}

impl<R: DataReader> DataStream<R> {
    /// Creates a new `DataStream` instance.
    ///
    /// # Arguments
    /// * `reader` - The data source to stream from (e.g., `File`, `TcpStream`).
    /// * `chunk_size` - The size of the internal buffer and the chunk to process.
    ///
    /// # Returns
    /// A `Result` containing a `DataStream` or an `InvalidChunkSize` error.
    pub fn new(reader: R, chunk_size: usize) -> Result<Self, DataStreamError> {
        if chunk_size == 0 {
            return Err(DataStreamError::InvalidChunkSize);
        }

        Ok(Self {
            reader,
            buffer: vec![0; chunk_size],
        })
    }

    /// Processes the data stream chunk by chunk, applying a closure that can return a `Result`.
    ///
    /// This is the core engine for processing data. It reads from the underlying reader
    /// into its internal buffer and then calls the provided processor with a slice
    /// of the data read. No additional data copies are made after the initial read.
    ///
    /// # Arguments
    /// * `processor` - A mutable closure that processes each chunk. It takes
    /// a byte slice of the chunk and returns a `Result`.
    pub fn process_chunks<F, E>(&mut self, mut processor: F) -> Result<(), DataStreamError>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
        E: Error + Into<String>,
    {
        loop {
            // Read a chunk of data from the reader into the internal buffer.
            let bytes_read = self.reader.read(&mut self.buffer)?;
            
            // If we've reached the end of the stream, break the loop.
            if bytes_read == 0 {
                break;
            }

            // Process the non-empty part of the buffer. This is where we achieve our "zero-copy"
            // philosophy, as we pass a slice directly to the processor without an extra allocation.
            if let Err(e) = processor(&self.buffer[..bytes_read]) {
                return Err(DataStreamError::ProcessorError(e.into()));
            }
        }
        Ok(())
    }

    /// Processes the data stream chunk by chunk, applying a simple closure to each chunk.
    ///
    /// This is a simplified version of `process_chunks` for cases where the processor
    /// does not need to return a `Result` or a custom error.
    ///
    /// # Arguments
    /// * `processor` - A mutable closure that processes each chunk. It takes
    /// a byte slice of the chunk.
    pub fn for_each_chunk<F>(&mut self, mut processor: F) -> Result<(), DataStreamError>
    where
        F: FnMut(&[u8]),
    {
        loop {
            // Read a chunk of data from the reader.
            let bytes_read = self.reader.read(&mut self.buffer)?;
            if bytes_read == 0 {
                break;
            }
            
            // Apply the processor function to the chunk. This is the "no copy" part.
            processor(&self.buffer[..bytes_read]);
        }
        Ok(())
    }
}

/// A convenience function to create a `DataStream` from a file path.
///
/// # Arguments
/// * `path` - The path to the file to stream.
/// * `chunk_size` - The size of the chunks to read.
///
/// # Returns
/// A `Result` containing a `DataStream` instance or a `DataStreamError`.
pub fn new_file_stream<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<DataStream<File>, DataStreamError> {
    let file = File::open(path)?;
    DataStream::new(file, chunk_size)
}

/// A convenience function to create a `DataStream` from a network stream.
///
/// # Arguments
/// * `stream` - The `TcpStream` to read from.
/// * `chunk_size` - The size of the chunks to read.
///
/// # Returns
/// A `Result` containing a `DataStream` instance or a `DataStreamError`.
pub fn new_network_stream(stream: TcpStream, chunk_size: usize) -> Result<DataStream<TcpStream>, DataStreamError> {
    DataStream::new(stream, chunk_size)
}