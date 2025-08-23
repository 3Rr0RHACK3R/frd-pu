// src/data_stream.rs

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::net::TcpStream;
use std::path::Path;

/// A trait for any type that can be read from in a chunked manner.
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
                write!(f, "Invalid chunk size. Must be greater than 0.")
            }
            DataStreamError::ProcessorError(msg) => write!(f, "Processor error: {}", msg),
        }
    }
}

/// A professional-grade, high-performance, and efficient data streaming engine.
///
/// This structure provides a way to read and process large data streams from
/// various sources (like files or network connections) in a chunked manner,
/// ensuring low memory footprint and high throughput.
///
/// # Arguments
///
/// * `R` - The type of the data source, which must implement the `DataReader` trait.
///
/// # Examples
///
/// ```
/// use frd_pu::data_stream::{DataStream, new_file_stream};
/// use std::fs::File;
/// use std::io::{self, Cursor};
///
/// // Create a dummy file stream. In a real application, you'd open a file.
/// let data = "This is a test stream of data.";
/// let dummy_reader = Cursor::new(data.as_bytes());
///
/// // Create a new `DataStream` from the dummy reader.
/// let mut stream = DataStream::new(dummy_reader, 10).unwrap();
///
/// // Process the stream, printing each chunk.
/// stream.for_each_chunk(|chunk| {
///     println!("Chunk: {:?}", String::from_utf8_lossy(chunk));
/// }).unwrap();
/// ```
pub struct DataStream<R: DataReader> {
    reader: R,
    buffer: Vec<u8>,
}

impl<R: DataReader> DataStream<R> {
    /// Creates a new `DataStream` with the specified chunk size.
    ///
    /// # Arguments
    /// * `reader` - The underlying data source to read from.
    /// * `chunk_size` - The size of each data chunk in bytes.
    ///
    /// # Returns
    /// A `Result` containing the `DataStream` instance or an `InvalidChunkSize` error.
    pub fn new(reader: R, chunk_size: usize) -> Result<Self, DataStreamError> {
        if chunk_size == 0 {
            return Err(DataStreamError::InvalidChunkSize);
        }
        Ok(DataStream {
            reader,
            buffer: vec![0; chunk_size],
        })
    }

    /// Processes the data stream chunk by chunk, applying a closure to each chunk.
    ///
    /// The processor closure can return a `Result` to signal errors. This is
    /// ideal for tasks like parsing, compression, or validation.
    ///
    /// # Arguments
    /// * `processor` - A mutable closure that processes each chunk. It takes
    /// a byte slice of the chunk and returns a `Result`.
    pub fn process_chunks<F, E>(&mut self, mut processor: F) -> Result<(), DataStreamError>
    where
        F: FnMut(&[u8]) -> Result<(), E>,
        E: Into<String>,
    {
        loop {
            // Read a chunk of data from the reader.
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

/// A convenience function to create a `DataStream` from a file path.
///
/// # Arguments
/// * `path` - The path to the file to stream.
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
///
/// # Returns
/// A `Result` containing a `DataStream` instance or a `DataStreamError`.
pub fn new_network_stream(stream: TcpStream, chunk_size: usize) -> Result<DataStream<TcpStream>, DataStreamError> {
    DataStream::new(stream, chunk_size)
}