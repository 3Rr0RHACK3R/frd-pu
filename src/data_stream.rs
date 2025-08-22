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
                write!(f, "Invalid chunk size: must be greater than 0.")
            }
            DataStreamError::ProcessorError(e) => write!(f, "Processor error during data stream: {}", e),
        }
    }
}

/// A struct for reading and processing a data stream in chunks.
///
/// This is a generic struct that can work with any type that implements the
/// `DataReader` trait, enabling it to process data from various sources.
pub struct DataStream<R: DataReader> {
    reader: R,
    buffer: Vec<u8>,
}

/// Creates a new data stream from a file.
///
/// This function is a convenience wrapper for creating a `DataStream` from a
/// file path. It is ideal for local file processing tasks.
///
/// # Arguments
/// * `path` - The path to the file to be streamed.
/// * `chunk_size` - The size of each chunk to read in bytes.
///
/// # Returns
/// A `Result` containing the new `DataStream` or a `DataStreamError`.
///
/// # Examples
///
/// ```no_run
/// use frd_pu::data_stream::{new_file_stream, DataStreamError};
/// use std::path::Path;
///
/// // Create a dummy file for this example.
/// // In a real scenario, you would use an existing file.
/// std::fs::write("example.txt", "Hello, world!").unwrap();
///
/// let path = Path::new("example.txt");
/// let mut file_stream = new_file_stream(path, 4).unwrap();
///
/// // Process the file in chunks.
/// let result = file_stream.process_chunks(|chunk| {
///     // Your processing logic here.
///     let chunk_str = String::from_utf8_lossy(chunk);
///     println!("Processed chunk: {}", chunk_str);
///     Ok(())
/// });
///
/// assert!(result.is_ok());
/// ```
pub fn new_file_stream<P: AsRef<Path>>(path: P, chunk_size: usize) -> Result<DataStream<File>, DataStreamError> {
    if chunk_size == 0 {
        return Err(DataStreamError::InvalidChunkSize);
    }
    let file = File::open(path)?;
    Ok(DataStream {
        reader: file,
        buffer: vec![0; chunk_size],
    })
}

/// Creates a new data stream from a network connection.
///
/// This function is a convenience wrapper for creating a `DataStream` from a
/// TCP stream. It is ideal for processing data from network sockets.
///
/// # Arguments
/// * `stream` - The `TcpStream` to be read from.
/// * `chunk_size` - The size of each chunk to read in bytes.
///
/// # Returns
/// A `Result` containing the new `DataStream` or a `DataStreamError`.
///
/// # Examples
///
/// ```no_run
/// use frd_pu::data_stream::{new_network_stream, DataStreamError};
/// use std::net::TcpStream;
///
/// // Note: This example requires a running server to connect to.
/// // Assuming a server is listening on 127.0.0.1:8080.
/// let stream = TcpStream::connect("127.0.0.1:8080").unwrap();
///
/// let mut network_stream = new_network_stream(stream, 128).unwrap();
///
/// let result = network_stream.process_chunks(|chunk| {
///     // Your processing logic for network data.
///     // This could be parsing a protocol, etc.
///     println!("Received chunk of size: {}", chunk.len());
///     Ok(())
/// });
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
            let bytes_read = self.reader.read(&mut self.buffer)?;
            if bytes_read == 0 {
                break; // End of stream
            }
            if let Err(e) = processor(&self.buffer[..bytes_read]) {
                return Err(DataStreamError::ProcessorError(e.into()));
            }
        }
        Ok(())
    }
}
