// src/data_stream.rs

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::net::TcpStream;
use std::path::Path;

/// A trait for any type that can be read from in a chunked manner.
/// This abstracts over different data sources like files or network streams.
trait DataReader: Read {}

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
                write!(f, "Invalid chunk size provided. Must be greater than 0.")
            }
            DataStreamError::ProcessorError(msg) => write!(f, "Data stream processor error: {}", msg),
        }
    }
}

/// A highly efficient, zero-copy data stream processor.
///
/// This struct is the core of the data streaming functionality. It handles reading
/// from any type that implements the `DataReader` trait in fixed-size chunks,
/// minimizing memory overhead and maximizing performance.
pub struct DataStream<R: DataReader> {
    reader: R,
    buffer: Vec<u8>,
}

/// Creates a new file-based data stream.
///
/// This function opens a file at the given path and prepares it for chunked reading.
/// It's perfect for processing massive files that won't fit entirely in memory.
///
/// # Arguments
/// * `path` - The path to the file to be streamed.
/// * `chunk_size` - The size of each chunk to read in bytes.
///
/// # Returns
/// A `Result` containing the `DataStream` or a `DataStreamError` if the file cannot be opened.
///
/// # Examples
///
/// ```
/// use frd_pu::data_stream::{new_file_stream, DataStreamError};
/// use std::path::Path;
/// use std::fs::File;
/// use std::io::Write;
///
/// // Create a dummy file for the example.
/// let path = Path::new("dummy_file.txt");
/// let mut file = File::create(path).unwrap();
/// file.write_all(b"Hello, this is a test file for the data stream.").unwrap();
///
/// // Create a new stream with a chunk size of 10 bytes.
/// let result = new_file_stream(path, 10);
///
/// assert!(result.is_ok());
/// ```
pub fn new_file_stream(path: &Path, chunk_size: usize) -> Result<DataStream<File>, DataStreamError> {
    if chunk_size == 0 {
        return Err(DataStreamError::InvalidChunkSize);
    }
    let file = File::open(path)?;
    Ok(DataStream {
        reader: file,
        buffer: vec![0; chunk_size],
    })
}

/// Creates a new network-based data stream.
///
/// This function wraps a `TcpStream` and prepares it for chunked reading. This is ideal
/// for processing incoming data from a network connection without loading the entire
/// stream into memory.
///
/// # Arguments
/// * `stream` - The `TcpStream` to be wrapped.
/// * `chunk_size` - The size of each chunk to read in bytes.
///
/// # Returns
/// A `Result` containing the `DataStream` or a `DataStreamError` if the chunk size is invalid.
///
/// # Examples
///
/// ```ignore
/// // This example is `ignore`d because it requires a network connection.
/// use frd_pu::data_stream::{new_network_stream, DataStreamError};
/// use std::net::TcpStream;
///
/// let stream = TcpStream::connect("127.0.0.1:8080").unwrap();
/// let result = new_network_stream(stream, 4096);
///
/// assert!(result.is_ok());
/// ```
pub fn new_network_stream(stream: TcpStream, chunk_size: usize) -> Result<DataStream<TcpStream>, DataStreamError> {
    if chunk_size == 0 {
        return Err(DataStreamError::InvalidChunkSize);
    }
    Ok(DataStream {
        reader: stream,
        buffer: vec![0; chunk_size],
    })
}

impl<R: DataReader> DataStream<R> {
    /// Processes the data stream chunk by chunk with a provided closure.
    ///
    /// This function reads the data in chunks and applies a given processor
    /// function to each chunk. This version is more flexible, allowing the
    /// processor to return a value or a custom error.
    ///
    /// # Arguments
    /// * `processor` - A mutable closure that processes each chunk. It takes
    /// a byte slice of the chunk and returns a `Result`.
    ///
    /// # Returns
    /// A `Result` indicating success or failure of the streaming operation.
    pub fn process_chunks<F, E>(&mut self, mut processor: F) -> Result<(), DataStreamError>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
        E: Into<String>,
    {
        loop {
            // Read a chunk of data from the reader into the buffer.
            let bytes_read = self.reader.read(&mut self.buffer)?;
            
            // If we've reached the end of the stream, break the loop.
            if bytes_read == 0 {
                break; // End of stream
            }

            // Process the non-empty part of the buffer.
            if let Err(e) = processor(&self.buffer[..bytes_read]) {
                // If the processor returns an error, map it to our DataStreamError.
                return Err(DataStreamError::ProcessorError(e.into()));
            }
        }
        Ok(())
    }

    /// Processes the data stream chunk by chunk, applying a simple closure to each chunk.
    ///
    /// This function is a simpler alternative to `process_chunks` for cases where the
    /// processor does not need to return a `Result` or a custom error.
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
                break; // End of stream
            }
            
            // Apply the processor function to the chunk.
            processor(&self.buffer[..bytes_read]);
        }
        Ok(())
    }
}
