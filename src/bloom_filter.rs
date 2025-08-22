// src/bloom_filter.rs

use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::f64;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Error type for Bloom filter creation.
#[derive(Debug, PartialEq)]
pub enum BloomFilterError {
    InvalidProbability,
    InvalidCapacity,
}

impl fmt::Display for BloomFilterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            BloomFilterError::InvalidProbability => write!(f, "Invalid false positive probability. Must be between 0.0 and 1.0 (exclusive)."),
            BloomFilterError::InvalidCapacity => write!(f, "Invalid capacity. Must be greater than 0."),
        }
    }
}

impl Error for BloomFilterError {}

/// A memory-efficient, probabilistic data structure.
pub struct BloomFilter {
    bit_vector: Vec<u8>,
    k: usize, // Number of hash functions
    m: usize, // Size of the bit vector
}

impl BloomFilter {
    /// Creates a new Bloom filter with a given capacity and false positive probability.
    pub fn new(capacity: usize, false_positive_probability: f64) -> Result<Self, BloomFilterError> {
        if !(0.0..1.0).contains(&false_positive_probability) {
            return Err(BloomFilterError::InvalidProbability);
        }
        if capacity == 0 {
            return Err(BloomFilterError::InvalidCapacity);
        }

        // Calculate the optimal number of bits (m)
        let m_f64 = (-1.0 * (capacity as f64) * false_positive_probability.ln()) / (2.0f64.ln().powi(2));
        let m = (m_f64.ceil() as usize);

        // Calculate the optimal number of hash functions (k)
        let k_f64 = (m_f64 / (capacity as f64)) * 2.0f64.ln();
        let k = (k_f64.ceil() as usize).max(1);

        // Ensure m is a power of 2 and a multiple of 8 for bitwise operations and byte-based storage
        // This is a crucial fix to avoid runtime panics and ensure proper functionality
        let m = ((m as u64).next_power_of_2() as usize).max(8);
        
        // The bit vector size in bytes
        let m_bytes = m / 8;

        Ok(BloomFilter {
            bit_vector: vec![0; m_bytes],
            k,
            m,
        })
    }

    /// Adds an item to the Bloom filter.
    pub fn add<T: Hash + ?Sized>(&mut self, item: &T) {
        for i in 0..self.k {
            let mut hasher = DefaultHasher::new();
            // This is a simple but effective way to generate different hashes
            // It relies on hashing both the loop index and the item
            (i, item).hash(&mut hasher);
            let hash = hasher.finish();
            
            // Custom bitwise modulo to replace math_utils
            let bit_index = (hash as usize) & (self.m - 1);
            
            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;
            self.bit_vector[byte_index] |= 1 << bit_offset;
        }
    }

    /// Checks if an item may be in the set.
    ///
    /// Returns `false` if the item is definitely not in the set.
    /// Returns `true` if the item is *probably* in the set.
    pub fn check<T: Hash + ?Sized>(&self, item: &T) -> bool {
        for i in 0..self.k {
            let mut hasher = DefaultHasher::new();
            (i, item).hash(&mut hasher);
            let hash = hasher.finish();

            // Custom bitwise modulo to replace math_utils
            let bit_index = (hash as usize) & (self.m - 1);

            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;
            if (self.bit_vector[byte_index] & (1 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }
}

// Re-export the public API for easy access.
pub use BloomFilter as new_bloom_filter;