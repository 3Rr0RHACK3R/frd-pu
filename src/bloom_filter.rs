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

        // Calculate optimal size (m) and number of hash functions (k)
        let m = (-1.0 * (capacity as f64) * false_positive_probability.ln() / (2.0f64.ln().powi(2))).ceil() as usize;
        let k = (m as f64 / capacity as f64 * 2.0f64.ln()).round() as usize;
        let byte_size = (m + 7) / 8; // Ceiling division

        Ok(BloomFilter {
            bit_vector: vec![0; byte_size],
            k: k.max(1), // Ensure at least one hash function
            m,
        })
    }

    /// Adds an item to the Bloom filter.
    pub fn add<T: Hash + ?Sized>(&mut self, item: &T) {
        for i in 0..self.k {
            let mut hasher = DefaultHasher::new();
            (i, item).hash(&mut hasher);
            let hash = hasher.finish();
            let bit_index = (hash % self.m as u64) as usize;
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
            let bit_index = (hash % self.m as u64) as usize;
            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;
            if (self.bit_vector[byte_index] & (1 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }
}

/// A convenience function to create a new `BloomFilter`.
pub fn new_bloom_filter(capacity: usize, false_positive_probability: f64) -> Result<BloomFilter, BloomFilterError> {
    BloomFilter::new(capacity, false_positive_probability)
}
