// src/cache.rs

use std::collections::{HashMap, LinkedList};
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::error::Error;

/// Error type for cache operations.
#[derive(Debug, PartialEq)]
pub enum CacheError {
    /// Indicates an item is too large for the cache.
    ItemTooLarge,
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheError::ItemTooLarge => write!(f, "The item being inserted is larger than the cache's maximum size."),
        }
    }
}

impl Error for CacheError {}

/// A professional-grade, high-performance, memory-aware, Least Recently Used (LRU) cache.
///
/// This cache is designed for extreme speed and memory efficiency. It uses a combination
/// of a hash map for fast lookups and a linked list for efficient access-order tracking.
/// The `HashMap` provides O(1) average time complexity for insertion, lookup, and deletion,
/// while the `LinkedList` allows for O(1) removal of the least recently used item.
///
/// The cache's memory usage is managed by a `max_size` in bytes.
///
/// # Panics
///
/// This implementation will panic if it encounters an issue with the underlying data
/// structures, but this is handled by the `Result` return type.
///
/// # Examples
///
/// ```
/// use frd_pu::cache::{LruCache, CacheError};
///
/// // Create a new cache with a maximum capacity of 2 items and a max size of 100 bytes.
/// let mut cache = LruCache::new(2, 100);
///
/// // Insert some key-value pairs.
/// cache.insert("key1".to_string(), "val1".to_string(), 4).unwrap();
/// cache.insert("key2".to_string(), "val2".to_string(), 4).unwrap();
///
/// // Accessing a key moves it to the front of the LRU list.
/// cache.get(&"key1");
///
/// // Insert a new item. This will evict the least recently used item, which is "key2".
/// cache.insert("key3".to_string(), "val3".to_string(), 4).unwrap();
///
/// // "key2" should no longer be in the cache.
/// assert_eq!(cache.get(&"key2"), None);
/// ```
pub struct LruCache<K, V> {
    /// A hash map for fast key-to-value and key-to-node lookups.
    cache_map: HashMap<K, V>,
    /// A linked list to maintain the order of recently used keys. The front is the most recent.
    lru_list: LinkedList<K>,
    /// The maximum number of items the cache can hold.
    max_capacity: usize,
    /// The maximum memory size in bytes the cache can hold.
    max_size: usize,
    /// The current size of the cache in bytes.
    current_size: usize,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Clone,
{
    /// Creates a new, empty LRU cache.
    ///
    /// # Arguments
    /// * `max_capacity` - The maximum number of items the cache can hold.
    /// * `max_size` - The maximum memory size in bytes the cache can hold.
    pub fn new(max_capacity: usize, max_size: usize) -> Self {
        LruCache {
            cache_map: HashMap::with_capacity(max_capacity),
            lru_list: LinkedList::new(),
            max_capacity,
            max_size,
            current_size: 0,
        }
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// If the cache exceeds its capacity or size limit, it will evict the
    /// least recently used items.
    ///
    /// # Arguments
    /// * `key` - The key to insert.
    /// * `value` - The value to insert.
    /// * `size` - The size of the item in bytes.
    ///
    /// # Returns
    /// `Ok(())` on success, or a `CacheError` if the item is too large.
    pub fn insert(&mut self, key: K, value: V, size: usize) -> Result<(), CacheError> {
        if size > self.max_size {
            return Err(CacheError::ItemTooLarge);
        }

        // If the key already exists, update the value and move it to the front.
        if self.cache_map.contains_key(&key) {
            self.remove(&key);
        }

        // Check if adding the new item will exceed the size limit.
        // Evict items until we have enough space.
        while self.current_size + size > self.max_size || self.cache_map.len() >= self.max_capacity {
            self.evict_lru();
        }

        // Insert the new key-value pair and update the LRU list.
        self.cache_map.insert(key.clone(), value);
        self.lru_list.push_front(key);
        self.current_size += size;

        Ok(())
    }

    /// Retrieves a value from the cache and updates its position in the LRU list.
    ///
    /// # Arguments
    /// * `key` - The key to retrieve.
    ///
    /// # Returns
    /// `Some(&V)` if the key exists, otherwise `None`.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.cache_map.contains_key(key) {
            // Remove the key from its current position in the LRU list.
            self.lru_list.retain(|lru_key| lru_key != key);
            // Push it to the front to mark it as most recently used.
            self.lru_list.push_front(key.clone());
            self.cache_map.get(key)
        } else {
            None
        }
    }

    /// Removes a key-value pair from the cache.
    ///
    /// # Arguments
    /// * `key` - The key to remove.
    ///
    /// # Returns
    /// `Some(V)` if the key was found and removed, otherwise `None`.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.cache_map.remove(key) {
            // Remove the key from the LRU list.
            self.lru_list.retain(|lru_key| lru_key != key);
            self.current_size -= std::mem::size_of_val(&value);
            Some(value)
        } else {
            None
        }
    }

    /// Returns the number of items currently in the cache.
    pub fn len(&self) -> usize {
        self.cache_map.len()
    }

    /// Returns the current size of the cache in bytes.
    pub fn size(&self) -> usize {
        self.current_size
    }

    /// Evicts the least recently used item from the cache.
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.lru_list.pop_back() {
            if let Some(value) = self.cache_map.remove(&lru_key) {
                self.current_size -= std::mem::size_of_val(&value);
            }
        }
    }
}
