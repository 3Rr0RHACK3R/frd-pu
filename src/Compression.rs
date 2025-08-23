// src/compression.rs

//! # High-Performance LZ77-Style Compression Engine
//!
//! A zero-dependency compression module implementing LZ77-style algorithm for optimal
//! performance and memory efficiency. Designed for the FRD-PU library's philosophy
//! of "doing more with less."
//!
//! ## Features:
//! * Pure Rust implementation with zero external dependencies
//! * LZ77-style sliding window compression
//! * Memory-efficient with configurable window sizes
//! * Fast compression and decompression
//! * Thread-safe operations

use std::fmt;

/// Maximum lookback window size for LZ77 compression
const MAX_WINDOW_SIZE: usize = 32768; // 32KB window
/// Maximum match length
const MAX_MATCH_LENGTH: usize = 258;
/// Minimum match length to be worth encoding
const MIN_MATCH_LENGTH: usize = 3;

/// Errors that can occur during compression operations
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionError {
    /// Input data is empty
    EmptyInput,
    /// Invalid compressed data format
    InvalidFormat,
    /// Compression failed due to internal error
    CompressionFailed,
    /// Decompression failed due to corrupted data
    DecompressionFailed,
    /// Window size is invalid
    InvalidWindowSize,
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionError::EmptyInput => write!(f, "Input data is empty"),
            CompressionError::InvalidFormat => write!(f, "Invalid compressed data format"),
            CompressionError::CompressionFailed => write!(f, "Compression operation failed"),
            CompressionError::DecompressionFailed => write!(f, "Decompression operation failed"),
            CompressionError::InvalidWindowSize => write!(f, "Invalid window size specified"),
        }
    }
}

impl std::error::Error for CompressionError {}

/// Represents a match found in the LZ77 sliding window
#[derive(Debug, Clone)]
struct Match {
    /// Distance back to the start of the match
    distance: u16,
    /// Length of the match
    length: u8,
}

/// High-performance LZ77 compression engine
pub struct CompressionEngine {
    window_size: usize,
}

impl Default for CompressionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionEngine {
    /// Create a new compression engine with default settings
    pub fn new() -> Self {
        Self {
            window_size: MAX_WINDOW_SIZE,
        }
    }

    /// Create a new compression engine with custom window size
    pub fn with_window_size(window_size: usize) -> Result<Self, CompressionError> {
        if window_size == 0 || window_size > MAX_WINDOW_SIZE {
            return Err(CompressionError::InvalidWindowSize);
        }

        Ok(Self { window_size })
    }

    /// Find the longest match in the sliding window
    fn find_longest_match(&self, data: &[u8], pos: usize) -> Option<Match> {
        if pos < MIN_MATCH_LENGTH {
            return None;
        }

        let search_start = pos.saturating_sub(self.window_size);
        let max_length = std::cmp::min(MAX_MATCH_LENGTH, data.len() - pos);
        
        let mut best_match: Option<Match> = None;
        let mut best_length = MIN_MATCH_LENGTH - 1;

        // Search backwards through the window
        for i in (search_start..pos).rev() {
            let mut length = 0;
            
            // Find match length
            while length < max_length
                && pos + length < data.len()
                && data[i + length] == data[pos + length]
            {
                length += 1;
            }

            if length >= MIN_MATCH_LENGTH && length > best_length {
                best_length = length;
                best_match = Some(Match {
                    distance: (pos - i) as u16,
                    length: length as u8,
                });

                // If we found a very long match, we can stop searching
                if length >= MAX_MATCH_LENGTH {
                    break;
                }
            }
        }

        best_match
    }

    /// Compress data using LZ77 algorithm
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            return Err(CompressionError::EmptyInput);
        }

        let mut compressed = Vec::with_capacity(data.len());
        let mut pos = 0;

        // Add magic header to identify compressed data
        compressed.extend_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]);
        
        // Add original size (4 bytes, little-endian)
        compressed.extend_from_slice(&(data.len() as u32).to_le_bytes());

        while pos < data.len() {
            if let Some(match_found) = self.find_longest_match(data, pos) {
                // Encode as match: [1][distance:2][length:1]
                compressed.push(1); // Match flag
                compressed.extend_from_slice(&match_found.distance.to_le_bytes());
                compressed.push(match_found.length);
                pos += match_found.length as usize;
            } else {
                // Encode as literal: [0][byte:1]
                compressed.push(0); // Literal flag
                compressed.push(data[pos]);
                pos += 1;
            }
        }

        Ok(compressed)
    }

    /// Decompress LZ77 compressed data
    pub fn decompress(&self, compressed: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if compressed.len() < 8 {
            return Err(CompressionError::InvalidFormat);
        }

        // Check magic header
        if &compressed[0..4] != &[0xFF, 0xFE, 0xFD, 0xFC] {
            return Err(CompressionError::InvalidFormat);
        }

        // Read original size
        let original_size = u32::from_le_bytes([
            compressed[4], compressed[5], compressed[6], compressed[7]
        ]) as usize;

        let mut decompressed = Vec::with_capacity(original_size);
        let mut pos = 8; // Skip header and size

        while pos < compressed.len() && decompressed.len() < original_size {
            if pos >= compressed.len() {
                return Err(CompressionError::DecompressionFailed);
            }

            let flag = compressed[pos];
            pos += 1;

            if flag == 0 {
                // Literal byte
                if pos >= compressed.len() {
                    return Err(CompressionError::DecompressionFailed);
                }
                decompressed.push(compressed[pos]);
                pos += 1;
            } else if flag == 1 {
                // Match: distance (2 bytes) + length (1 byte)
                if pos + 2 >= compressed.len() {
                    return Err(CompressionError::DecompressionFailed);
                }

                let distance = u16::from_le_bytes([compressed[pos], compressed[pos + 1]]) as usize;
                let length = compressed[pos + 2] as usize;
                pos += 3;

                if distance == 0 || distance > decompressed.len() {
                    return Err(CompressionError::DecompressionFailed);
                }

                let start_pos = decompressed.len() - distance;
                
                // Copy bytes (handle overlapping copies correctly)
                for i in 0..length {
                    if decompressed.len() >= original_size {
                        break;
                    }
                    let byte_to_copy = decompressed[start_pos + i];
                    decompressed.push(byte_to_copy);
                }
            } else {
                return Err(CompressionError::DecompressionFailed);
            }
        }

        if decompressed.len() != original_size {
            return Err(CompressionError::DecompressionFailed);
        }

        Ok(decompressed)
    }

    /// Get compression ratio for given data
    pub fn estimate_compression_ratio(&self, data: &[u8]) -> Result<f64, CompressionError> {
        let compressed = self.compress(data)?;
        Ok(compressed.len() as f64 / data.len() as f64)
    }
}

/// Compress data using default settings
pub fn compress_data(data: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let engine = CompressionEngine::new();
    engine.compress(data)
}

/// Decompress previously compressed data
pub fn decompress_data(compressed: &[u8]) -> Result<Vec<u8>, CompressionError> {
    let engine = CompressionEngine::new();
    engine.decompress(compressed)
}

/// Compress text data (convenience function)
pub fn compress_text(text: &str) -> Result<Vec<u8>, CompressionError> {
    compress_data(text.as_bytes())
}

/// Decompress to text data (convenience function)
pub fn decompress_to_text(compressed: &[u8]) -> Result<String, CompressionError> {
    let decompressed = decompress_data(compressed)?;
    String::from_utf8(decompressed).map_err(|_| CompressionError::DecompressionFailed)
}

/// Calculate compression statistics
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub space_saved: usize,
    pub space_saved_percentage: f64,
}

impl CompressionStats {
    pub fn new(original_size: usize, compressed_size: usize) -> Self {
        let compression_ratio = compressed_size as f64 / original_size as f64;
        let space_saved = original_size.saturating_sub(compressed_size);
        let space_saved_percentage = (space_saved as f64 / original_size as f64) * 100.0;

        Self {
            original_size,
            compressed_size,
            compression_ratio,
            space_saved,
            space_saved_percentage,
        }
    }
}

impl fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Original: {} bytes, Compressed: {} bytes, Ratio: {:.2}, Saved: {} bytes ({:.1}%)",
            self.original_size,
            self.compressed_size,
            self.compression_ratio,
            self.space_saved,
            self.space_saved_percentage
        )
    }
}

/// Get detailed compression statistics
pub fn get_compression_stats(data: &[u8]) -> Result<CompressionStats, CompressionError> {
    let compressed = compress_data(data)?;
    Ok(CompressionStats::new(data.len(), compressed.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compression() {
        let data = b"Hello, World! Hello, World! This is a test.";
        let compressed = compress_data(data).unwrap();
        let decompressed = decompress_data(&compressed).unwrap();
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    fn test_repetitive_data() {
        let data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let compressed = compress_data(data).unwrap();
        let decompressed = decompress_data(&compressed).unwrap();
        assert_eq!(data, decompressed.as_slice());
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_text_compression() {
        let text = "This is a test string. This is a test string. Compression should work well on repetitive text.";
        let compressed = compress_text(text).unwrap();
        let decompressed = decompress_to_text(&compressed).unwrap();
        assert_eq!(text, decompressed);
    }

    #[test]
    fn test_empty_input() {
        let result = compress_data(&[]);
        assert_eq!(result.unwrap_err(), CompressionError::EmptyInput);
    }

    #[test]
    fn test_compression_stats() {
        let data = b"Hello, World! Hello, World! Hello, World!";
        let stats = get_compression_stats(data).unwrap();
        assert!(stats.compression_ratio < 1.0);
        assert!(stats.space_saved > 0);
    }
}