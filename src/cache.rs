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

/// A high-performance, memory-aware, Least Recently Used (LRU) cache.
///
/// This cache is designed for extreme speed and memory efficiency, adhering to the
/// FRD-PU philosophy of zero dependencies. It uses a combination of a hash map
/// for fast lookups and a linked list for efficient access-order tracking.
///
/// The cache's memory usage is managed by a `max_size` in bytes.
///
/// # Examples
///
/// ```
/// use frd_pu::cache::LruCache;
///
/// let mut cache = LruCache::new(2, 100);
/// cache.insert("key1".to_string(), "val1".to_string(), 4).unwrap();
/// cache.insert("key2".to_string(), "val2".to_string(), 4).unwrap();
///
/// // Accessing "key1" makes it the most recently used.
/// cache.get(&"key1".to_string());
///
/// // Inserting "key3" will evict "key2", the least recently used item.
/// cache.insert("key3".to_string(), "val3".to_string(), 4).unwrap();
///
/// assert!(cache.get(&"key1".to_string()).is_some());
/// assert!(cache.get(&"key2".to_string()).is_none());
/// assert!(cache.get(&"key3".to_string()).is_some());
/// ```
pub struct LruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    cache_map: HashMap<K, V>,
    lru_list: LinkedList<K>,
    capacity: usize,
    max_size: usize,
    current_size: usize,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Debug,
{
    /// Creates a new `LruCache` with a specified capacity and maximum memory size.
    ///
    /// # Arguments
    /// * `capacity` - The maximum number of items the cache can hold. A value of 0 means
    ///   the cache will use a default capacity of 1024 items.
    /// * `max_size_bytes` - The maximum memory in bytes the cache can use. A value of 0 means
    ///   the cache will use a default of 1GB (1,073,741,824 bytes).
    pub fn new(capacity: usize, max_size_bytes: usize) -> Self {
        let capacity = if capacity == 0 { 1024 } else { capacity };
        let max_size = if max_size_bytes == 0 { 1_073_741_824 } else { max_size_bytes };
        
        LruCache {
            cache_map: HashMap::with_capacity(capacity),
            lru_list: LinkedList::new(),
            capacity,
            max_size,
            current_size: 0,
        }
    }

    /// Inserts a key-value pair into the cache.
    ///
    /// If the cache already contains the key, its value is updated and the item's
    /// position in the LRU list is refreshed.
    ///
    /// If inserting a new item causes the cache to exceed its capacity or maximum
    /// memory size, the least recently used item(s) are evicted.
    ///
    /// # Arguments
    /// * `key` - The key to insert.
    /// * `value` - The value to insert.
    /// * `size_in_bytes` - The memory size of the value in bytes.
    ///
    /// # Returns
    /// `Ok(())` on success, or a `CacheError` if the item is too large.
    ///
    /// # Examples
    ///
    /// ```
    /// use frd_pu::cache::LruCache;
    /// use frd_pu::cache::CacheError;
    ///
    /// let mut cache = LruCache::new(10, 1024);
    ///
    /// // Example: Insert and get an item
    /// let key = "test_key";
    /// let value = "test_value";
    /// let size = value.len();
    ///
    /// assert!(cache.insert(key.to_string(), value.to_string(), size).is_ok());
    /// assert_eq!(cache.get(&key.to_string()), Some(&value.to_string()));
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Example: Update an item
    /// let new_value = "new_value";
    /// let new_size = new_value.len();
    /// assert!(cache.insert(key.to_string(), new_value.to_string(), new_size).is_ok());
    /// assert_eq!(cache.get(&key.to_string()), Some(&new_value.to_string()));
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Example: Item too large
    /// let mut cache = LruCache::new(10, 10);
    /// let large_value = "this is a long value that is bigger than 10 bytes".to_string();
    /// let result = cache.insert("large_key".to_string(), large_value, 50);
    /// assert!(result.is_err());
    /// assert_eq!(result.unwrap_err(), CacheError::ItemTooLarge);
    /// ```
    pub fn insert(&mut self, key: K, value: V, size_in_bytes: usize) -> Result<(), CacheError> {
        // First, check if the item is too large for the cache.
        if size_in_bytes > self.max_size {
            return Err(CacheError::ItemTooLarge);
        }

        // If the key already exists, update its value and its position in the LRU list.
        if self.cache_map.contains_key(&key) {
            // Remove the old entry and its size from the cache.
            self.lru_list.retain(|lru_key| lru_key != &key);
            if let Some(old_value) = self.cache_map.remove(&key) {
                self.current_size -= std::mem::size_of_val(&old_value);
            }
        }
        
        // Add the new key-value pair.
        self.cache_map.insert(key.clone(), value);
        self.lru_list.push_front(key);
        self.current_size += size_in_bytes;

        // Evict items if capacity or size limits are exceeded.
        while self.cache_map.len() > self.capacity || self.current_size > self.max_size {
            self.evict_least_recent();
        }

        Ok(())
    }

    /// Retrieves a reference to the value associated with the key and updates its
    /// position in the LRU list.
    ///
    /// # Returns
    /// An `Option` containing a reference to the value, or `None` if the key
    /// is not in the cache.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.cache_map.contains_key(key) {
            // This is an inefficient way to update the list, but it's a simple, dependency-free
            // solution. For a truly professional library, a more complex data structure
            // (like a HashMap pointing to a DoublyLinkedList) would be used.
            self.lru_list.retain(|lru_key| lru_key != key);
            self.lru_list.push_front(key.clone());
            self.cache_map.get(key)
        } else {
            None
        }
    }

    /// Removes a key-value pair from the cache.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.cache_map.remove(key) {
            // Remove the key from the LRU list and update the size.
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
    fn evict_least_recent(&mut self) {
        if let Some(key) = self.lru_list.pop_back() {
            if let Some(value) = self.cache_map.remove(&key) {
                self.current_size -= std::mem::size_of_val(&value);
            }
        }
    }
}
