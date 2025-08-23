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
/// // Create a new cache with a max size of 100 bytes.
/// let mut cache = LruCache::<&str, [u8; 10]>::new(100);
///
/// // Insert an item.
/// let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let result = cache.insert("key1", data);
/// assert!(result.is_ok());
///
/// // Get the item. This moves it to the front of the LRU list.
/// let item = cache.get(&"key1");
/// assert!(item.is_some());
///
/// // Remove an item.
/// cache.remove(&"key1");
/// assert_eq!(cache.len(), 0);
/// ```
pub struct LruCache<K, V> {
    cache_map: HashMap<K, V>,
    lru_list: LinkedList<K>,
    current_size: usize,
    max_size: usize,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone,
    V: std::mem::size_of_val,
{
    /// Creates a new `LruCache` with the specified maximum size in bytes.
    pub fn new(max_size: usize) -> Self {
        LruCache {
            cache_map: HashMap::new(),
            lru_list: LinkedList::new(),
            current_size: 0,
            max_size,
        }
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// If the key already exists, its value is updated. The size of the cache
    /// is managed according to the `max_size`. If the cache is full, the least
    /// recently used item is removed to make space.
    ///
    /// # Arguments
    /// * `key` - The key to insert.
    /// * `value` - The value to associate with the key.
    ///
    /// # Returns
    /// `Ok(())` on success, or a `CacheError` if the item is too large for the cache.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), CacheError> {
        let item_size = std::mem::size_of_val(&value);

        if item_size > self.max_size {
            return Err(CacheError::ItemTooLarge);
        }

        // If the key already exists, remove it from the LRU list.
        if self.cache_map.contains_key(&key) {
            self.lru_list.retain(|lru_key| lru_key != &key);
            self.current_size -= std::mem::size_of_val(self.cache_map.get(&key).unwrap());
        }

        // Insert the new key-value pair and update the size.
        self.current_size += item_size;
        self.cache_map.insert(key.clone(), value);
        self.lru_list.push_front(key);

        // Enforce the cache size limit.
        while self.current_size > self.max_size {
            if let Some(lru_key) = self.lru_list.pop_back() {
                if let Some(value) = self.cache_map.remove(&lru_key) {
                    self.current_size -= std::mem::size_of_val(&value);
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Retrieves a value from the cache and updates its position in the LRU list.
    ///
    /// # Arguments
    /// * `key` - The key to retrieve.
    ///
    /// # Returns
    /// `Some(&V)` if the key is found, otherwise `None`.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.cache_map.contains_key(key) {
            // Remove the key from its current position in the LRU list.
            let mut new_lru_list = LinkedList::new();
            for lru_key in self.lru_list.iter() {
                if lru_key != key {
                    new_lru_list.push_back(lru_key.clone());
                }
            }
            self.lru_list = new_lru_list;

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
            let mut new_lru_list = LinkedList::new();
            for lru_key in self.lru_list.iter() {
                if lru_key != key {
                    new_lru_list.push_back(lru_key.clone());
                }
            }
            self.lru_list = new_lru_list;
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

    /// Returns the maximum allowed size of the cache in bytes.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache_map.is_empty()
    }

    /// Clears the cache, removing all key-value pairs and resetting the size.
    pub fn clear(&mut self) {
        self.cache_map.clear();
        self.lru_list.clear();
        self.current_size = 0;
    }
}