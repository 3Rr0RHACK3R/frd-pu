// src/hasher.rs

use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
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
/// let data = b"FRD-PU is the GOAT.";
/// let hash = hash_bytes(data);
/// assert_ne!(hash, 0); // The hash should not be zero for non-empty data.
/// ```
pub fn hash_bytes<T: Hash + ?Sized>(data: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

/// Hashes the content of a file specified by its path.
///
/// This function is a convenience wrapper around `hash_stream`. It opens
/// the file and hashes its content in a memory-efficient manner.
///
/// # Arguments
/// * `path` - The path to the file to be hashed.
///
/// # Returns
/// A `Result` containing the 64-bit hash or a `HasherError` if an I/O error occurs.
///
/// # Examples
///
/// ```
/// use frd_pu::hasher::{hash_file, HasherError};
/// use std::fs::File;
/// use std::io::Write;
/// use std::path::Path;
///
/// // Create a temporary file for the example.
/// let temp_dir = tempfile::tempdir().unwrap();
/// let file_path = temp_dir.path().join("my_file.txt");
///
/// {
///     let mut file = File::create(&file_path).unwrap();
///     write!(file, "FRD-PU is the GOAT.").unwrap();
/// }
///
/// let result = hash_file(&file_path);
/// assert!(result.is_ok());
/// ```
pub fn hash_file<P: AsRef<Path>>(path: P) -> Result<u64, HasherError> {
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
        let bytes_read = reader.read(&mut buffer).map_err(|e| HasherError::IoError(e.to_string()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.write(&buffer[..bytes_read]);
    }

    Ok(hasher.finish())
}