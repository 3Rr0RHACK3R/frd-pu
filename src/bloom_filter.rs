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

        // Calculate the optimal size of the bit vector (m) and the number of hash functions (k).
        // m = -(n * ln(p)) / (ln(2)^2)
        let m = (-(capacity as f64) * false_positive_probability.ln() / f64::consts::LN_2.powi(2)).ceil() as usize;
        // k = (m/n) * ln(2)
        let k = ((m as f64) / (capacity as f64) * f64::consts::LN_2).ceil() as usize;

        // Ensure m is at least 1 to avoid an empty bit vector.
        let m = m.max(1);
        let k = k.max(1);

        Ok(Self {
            bit_vector: vec![0; m / 8 + 1],
            k,
            m,
        })
    }

    /// Adds an item to the set by setting the corresponding bits in the bit vector.
    ///
    /// The bit indices are determined by running the item through `k` different hash functions.
    ///
    /// # Arguments
    /// * `item` - The item to add to the filter. It must implement the `Hash` trait.
    pub fn add<T: Hash + ?Sized>(&mut self, item: &T) {
        for i in 0..self.k {
            let mut hasher = DefaultHasher::new();
            // This is a simple but effective way to generate different hashes.
            // It relies on hashing both the loop index and the item.
            (i, item).hash(&mut hasher);
            let hash = hasher.finish();
            
            // We use a modulo operation to get the correct bit index.
            // The previous bitwise AND was incorrect because 'm' is not guaranteed to be a power of 2.
            let bit_index = (hash as usize) % self.m;
            
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

            // We use a modulo operation to get the correct bit index.
            // The previous bitwise AND was incorrect because 'm' is not guaranteed to be a power of 2.
            let bit_index = (hash as usize) % self.m;

            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;
            if (self.bit_vector[byte_index] & (1 << bit_offset)) == 0 {
                // If any of the bits are not set, the item is definitely not in the filter.
                return false;
            }
        }

        // If all bits are set, the item is probably in the filter.
        true
    }
}


