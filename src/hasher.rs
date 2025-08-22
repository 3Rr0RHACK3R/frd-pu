// src/hasher.rs

use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, Read};
use std::path::Path;

/// Error type for hashing operations.
#[derive(Debug, PartialEq)]
pub enum HasherError {
    /// Indicates an I/O error occurred during a file read operation.
    IoError(String),
}

impl fmt::Display for HasherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HasherError::IoError(msg) => write!(f, "I/O error during hashing: {}", msg),
        }
    }
}

impl Error for HasherError {}

/// Hashes a byte slice into a 64-bit integer.
///
/// This function uses the `DefaultHasher`, which is a fast, non-cryptographic
/// hash function ideal for use in hash maps and general-purpose hashing
/// where collision resistance is not the primary goal.
///
/// # Arguments
/// * `data` - The byte slice to be hashed.
///
/// # Returns
/// The resulting 64-bit hash as a `u64`.
///
/// # Examples
/// ```
/// use frd_pu::hasher::hash_bytes;
///
/// let data = b"Hello, world!";
/// let hash = hash_bytes(data);
/// assert_eq!(hash, 13735071190989083510); // A known hash value for this data
/// ```
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Hashes the content of a file in a memory-efficient, streaming manner.
///
/// This function reads the file in chunks to avoid loading the entire
/// content into memory at once. This makes it suitable for very large files.
///
/// # Arguments
/// * `path` - The path to the file to be hashed.
///
/// # Returns
/// A `Result` containing the 64-bit hash or a `HasherError` if an I/O error occurs.
///
/// # Examples
///
/// ```no_run
/// use frd_pu::hasher::hash_file;
/// use std::path::Path;
///
/// // Create a dummy file for this example.
/// // In a real scenario, you would use an existing file.
/// std::fs::write("example_file.txt", "This is some test data.").unwrap();
///
/// let path = Path::new("example_file.txt");
/// let result = hash_file(path);
/// assert!(result.is_ok());
/// ```
pub fn hash_file<P: AsRef<Path>>(path: P) -> Result<u64, HasherError> {
    // Open the file and wrap in a generic stream for processing.
    let file = File::open(path).map_err(|e| HasherError::IoError(e.to_string()))?;
    hash_stream(file)
}

/// Hashes data from any type that implements the `Read` trait.
///
/// This is the core, generic hashing function. It allows for hashing data
/// from any source, such as files, network streams, or byte buffers. It reads
/// the data in small chunks to ensure low memory usage.
///
/// # Arguments
/// * `reader` - A type that implements the `Read` trait.
///
/// # Returns
/// A `Result` containing the 64-bit hash or a `HasherError` if an I/O error occurs.
///
/// # Examples
///
/// ```
/// use frd_pu::hasher::hash_stream;
/// use std::io::Cursor;
///
/// // Use `Cursor` to simulate a stream from an in-memory byte slice.
/// let data = b"FRD-PU is the GOAT.";
/// let reader = Cursor::new(data);
///
/// let result = hash_stream(reader);
/// assert!(result.is_ok());
/// ```
pub fn hash_stream<R: Read>(mut reader: R) -> Result<u64, HasherError> {
    // We use a small buffer to read chunks of data.
    let mut buffer = [0; 4096];
    let mut hasher = DefaultHasher::new();

    loop {
        // Read the next chunk of data.
        let bytes_read = reader.read(&mut buffer).map_err(|e| HasherError::IoError(e.to_string()))?;

        if bytes_read == 0 {
            break; // End of the stream.
        }

        // Hash the chunk and continue the process.
        hasher.write(&buffer[..bytes_read]);
    }

    Ok(hasher.finish())
}
