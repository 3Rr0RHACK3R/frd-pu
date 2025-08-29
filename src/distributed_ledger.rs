//! # distributed_ledger.rs
//!
//! A high-performance, zero-dependency implementation of a distributed ledger.
//! This module provides the core logic for a lightweight, append-only record system
//! designed for speed and data integrity without the overhead of traditional
//! blockchain technologies. It is ideal for applications requiring a verifiable,
//! immutable chain of data, such as audit logs, supply chain tracking, or
//! secure transaction records.
//!
//! The design philosophy is centered on minimal resource consumption, high
//! throughput, and absolute reliability. This module leverages a linked list
//! of blocks, each cryptographically secured by its own hash and the hash of
//! the preceding block, ensuring data immutability and tamper-resistance.

use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::{Hasher, Hash};
use std::collections::hash_map::DefaultHasher;
use std::time::{SystemTime, UNIX_EPOCH};
use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// A custom, comprehensive error type for all ledger-related operations.
#[derive(Debug, PartialEq, Clone)]
pub enum LedgerError {
    /// Indicates that a block is invalid, typically due to a hash mismatch
    /// or a broken link to the previous block.
    InvalidBlock,
    /// Indicates a critical failure during the hashing process.
    HashGenerationFailure,
    /// Indicates that a record's data is invalid or in an unsupported format.
    InvalidRecordData,
    /// Indicates that the system time could not be retrieved.
    SystemTimeError,
}

impl Display for LedgerError {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match *self {
            LedgerError::InvalidBlock => write!(f, "Ledger Error: The block is invalid, failed hash verification."),
            LedgerError::HashGenerationFailure => write!(f, "Ledger Error: A critical failure occurred during hash generation."),
            LedgerError::InvalidRecordData => write!(f, "Ledger Error: The provided record data is invalid."),
            LedgerError::SystemTimeError => write!(f, "Ledger Error: Could not retrieve the current system time."),
        }
    }
}

impl Error for LedgerError {}

/// Represents a single immutable record or transaction within a block.
///
/// Each record contains a timestamp and data that can be any type
/// implementing the `Hash` and `Clone` traits.
#[derive(Debug, Clone)]
pub struct Record<T> {
    data: T,
    timestamp: u64,
}

/// The hash of a record is a combination of its data and its timestamp,
/// ensuring a unique and verifiable identifier.
impl<T: Hash> Hash for Record<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
        self.timestamp.hash(state);
    }
}

impl<T: Hash + Clone> Record<T> {
    /// Creates a new `Record` instance with a fresh timestamp from the system clock.
    ///
    /// # Arguments
    /// * `data` - The data payload for the record.
    ///
    /// # Errors
    /// Returns `LedgerError::SystemTimeError` if the system time is unavailable.
    pub fn new(data: T) -> Result<Self, LedgerError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| LedgerError::SystemTimeError)?
            .as_secs();
        Ok(Self { data, timestamp })
    }
}

/// A `Block` is the fundamental building block of the ledger.
///
/// It contains a unique ID, a list of records, a timestamp, and a cryptographic
/// hash that links it to the previous block in the chain.
#[derive(Debug, Clone)]
pub struct Block<T> {
    /// A monotonically increasing unique identifier for the block.
    id: u64,
    /// The timestamp of the block's creation.
    timestamp: u64,
    /// The vector of records contained within this block.
    records: Vec<Record<T>>,
    /// The hash of the previous block in the chain, ensuring integrity.
    previous_hash: u64,
    /// The cryptographic hash of the current block.
    hash: u64,
}

impl<T: Hash + Clone> Block<T> {
    /// Creates a new `Block` with the provided records and generates its
    /// unique cryptographic hash based on its contents.
    ///
    /// # Arguments
    /// * `id` - The unique ID for the new block.
    /// * `previous_hash` - The hash of the block that this new block will follow.
    /// * `records` - The data records to be stored in the new block.
    ///
    /// # Errors
    /// Returns `LedgerError` if the system time is unavailable or hashing fails.
    pub fn new(id: u64, previous_hash: u64, records: Vec<Record<T>>) -> Result<Self, LedgerError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| LedgerError::SystemTimeError)?
            .as_secs();

        let mut block = Block {
            id,
            timestamp,
            records,
            previous_hash,
            hash: 0,
        };

        // Calculate and set the block's hash.
        block.hash = block.calculate_hash()?;
        Ok(block)
    }

    /// Calculates the unique hash of the block based on all its constituent data.
    ///
    /// This function is deterministic and will always produce the same hash for the
    /// same block contents, which is critical for chain validation.
    ///
    /// # Errors
    /// Returns `LedgerError::HashGenerationFailure` if the hashing process fails.
    fn calculate_hash(&self) -> Result<u64, LedgerError> {
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);
        self.previous_hash.hash(&mut hasher);
        for record in &self.records {
            record.hash(&mut hasher);
        }
        Ok(hasher.finish())
    }

    /// Performs a self-validation check on the block's hash.
    ///
    /// # Errors
    /// Returns `LedgerError::InvalidBlock` if the calculated hash does not
    /// match the block's stored hash.
    pub fn validate_self(&self) -> Result<(), LedgerError> {
        let calculated_hash = self.calculate_hash()?;
        if calculated_hash != self.hash {
            return Err(LedgerError::InvalidBlock);
        }
        Ok(())
    }

    /// Accessor for the block's unique ID.
    pub fn get_id(&self) -> u64 { self.id }

    /// Accessor for the block's timestamp.
    pub fn get_timestamp(&self) -> u64 { self.timestamp }

    /// Accessor for the block's hash.
    pub fn get_hash(&self) -> u64 { self.hash }

    /// Accessor for the block's previous hash.
    pub fn get_previous_hash(&self) -> u64 { self.previous_hash }

    /// Accessor for the records contained within the block.
    pub fn get_records(&self) -> &Vec<Record<T>> { &self.records }
}

/// The main `Ledger` struct, which manages the chain of `Block`s.
///
/// It provides a simple, robust, and safe API for building and maintaining
/// an immutable data ledger.
pub struct Ledger<T> {
    /// A unique identifier for this specific ledger instance.
    id: u64,
    /// The core data structure: a vector representing the chain of blocks.
    chain: Vec<Block<T>>,
    /// An atomic counter to ensure block IDs are always unique and sequential.
    block_counter: AtomicU64,
}

impl<T: Hash + Clone> Ledger<T> {
    /// Creates a new `Ledger` and initializes it with a genesis block.
    ///
    /// The genesis block is the first block in the chain and has an ID of 0
    /// and a previous hash of 0.
    ///
    /// # Errors
    /// Returns `LedgerError` if the genesis block cannot be created.
    pub fn new(id: u64) -> Result<Self, LedgerError> {
        let genesis_block = Block::new(0, 0, vec![])?;
        Ok(Self {
            id,
            chain: vec![genesis_block],
            block_counter: AtomicU64::new(1),
        })
    }

    /// Adds a new block to the end of the ledger chain with the given records.
    ///
    /// The function automatically links the new block to the last block in the
    /// chain and validates its integrity before addition.
    ///
    /// # Arguments
    /// * `records` - A vector of records to be included in the new block.
    ///
    /// # Errors
    /// Returns `LedgerError::InvalidBlock` if the new block fails validation
    /// or if the previous hash link is broken.
    pub fn add_block(&mut self, records: Vec<Record<T>>) -> Result<(), LedgerError> {
        let last_block = self.chain.last().ok_or(LedgerError::InvalidBlock)?;
        let new_id = self.block_counter.load(Ordering::Relaxed);
        let new_block = Block::new(new_id, last_block.get_hash(), records)?;

        // Perform a quick self-validation before adding.
        new_block.validate_self()?;

        // Ensure the new block correctly links to the last one.
        if new_block.get_previous_hash() != last_block.get_hash() {
            return Err(LedgerError::InvalidBlock);
        }

        self.chain.push(new_block);
        self.block_counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Validates the integrity of the entire ledger chain.
    ///
    /// This function iterates through every block and verifies that its hash is
    /// correct and that it correctly links to the previous block.
    ///
    /// # Errors
    /// Returns `LedgerError::InvalidBlock` at the first sign of a corrupted block
    /// or a broken chain link.
    pub fn validate_chain(&self) -> Result<(), LedgerError> {
        for i in 1..self.chain.len() {
            let current_block = &self.chain[i];
            let previous_block = &self.chain[i - 1];

            // Verify the current block's hash.
            if current_block.validate_self().is_err() {
                return Err(LedgerError::InvalidBlock);
            }

            // Verify the link to the previous block.
            if current_block.get_previous_hash() != previous_block.get_hash() {
                return Err(LedgerError::InvalidBlock);
            }
        }
        Ok(())
    }

    /// Returns a reference to the last block in the ledger chain.
    pub fn get_last_block(&self) -> Option<&Block<T>> {
        self.chain.last()
    }

    /// Returns a reference to a block at a specific index.
    pub fn get_block_by_index(&self, index: usize) -> Option<&Block<T>> {
        self.chain.get(index)
    }

    /// Returns the total number of blocks in the ledger.
    pub fn get_length(&self) -> usize {
        self.chain.len()
    }
}
