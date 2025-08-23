// src/concurrent.rs

use std::fmt::Debug;
use std::sync::Mutex;
use std::error::Error;
use std::fmt;
use std::clone::Clone;

/// Error type for concurrent list operations.
#[derive(Debug, PartialEq)]
pub enum ConcurrentListError {
    /// Indicates that a lock could not be acquired.
    LockAcquisitionError,
}

impl fmt::Display for ConcurrentListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConcurrentListError::LockAcquisitionError => write!(f, "Failed to acquire mutex lock."),
        }
    }
}

impl Error for ConcurrentListError {}

/// A thread-safe list that allows for safe concurrent access from multiple threads.
///
/// This data structure wraps a standard `Vec` in a `Mutex` to provide exclusive
/// access and uses `Arc` to allow the list to be shared and passed between threads.
///
/// # Panics
///
/// This implementation will panic if the inner `Mutex` is poisoned by a thread
/// that panicked while holding the lock. In production code, you might want to
/// handle this more gracefully.
///
/// # Examples
///
/// ```
/// use frd_pu::concurrent::ConcurrentList;
/// use std::thread;
/// use std::sync::Arc;
///
/// // Create a new thread-safe list.
/// let list = Arc::new(ConcurrentList::new());
///
/// let mut handles = vec![];
///
/// for i in 0..10 {
///     let list_clone = Arc::clone(&list);
///     let handle = thread::spawn(move || {
///         // Push a number into the list.
///         list_clone.push(i).unwrap();
///     });
///     handles.push(handle);
/// }
///
/// for handle in handles {
///     handle.join().unwrap();
/// }
///
/// // Check the length of the list.
/// assert_eq!(list.len().unwrap(), 10);
///
/// // Retrieve the first item.
/// let item = list.get(0).unwrap();
/// assert!(item.is_some());
/// ```
#[derive(Debug, Default)]
pub struct ConcurrentList<T> {
    inner: Mutex<Vec<T>>,
}

impl<T> ConcurrentList<T> {
    /// Creates a new, empty `ConcurrentList`.
    pub fn new() -> Self {
        ConcurrentList {
            inner: Mutex::new(Vec::new()),
        }
    }

    /// Pushes an item to the end of the list in a thread-safe manner.
    ///
    /// # Arguments
    /// * `item` - The item to push.
    ///
    /// # Returns
    /// A `Result` indicating success or a `ConcurrentListError` if the lock failed.
    pub fn push(&self, item: T) -> Result<(), ConcurrentListError> {
        let mut list = self.inner.lock().map_err(|_| ConcurrentListError::LockAcquisitionError)?;
        list.push(item);
        Ok(())
    }

    /// Pops the last item from the list in a thread-safe manner.
    ///
    /// # Returns
    /// A `Result` containing an `Option` with the item, or an error if the lock failed.
    pub fn pop(&self) -> Result<Option<T>, ConcurrentListError> {
        let mut list = self.inner.lock().map_err(|_| ConcurrentListError::LockAcquisitionError)?;
        Ok(list.pop())
    }

    /// Returns a reference to the item at the specified index in a thread-safe manner.
    ///
    /// # Arguments
    /// * `index` - The index of the item to retrieve.
    ///
    /// # Returns
    /// A `Result` containing an `Option` with a reference to the item, or an error if the lock failed.
    pub fn get(&self, index: usize) -> Result<Option<T>, ConcurrentListError> where T: Clone {
        let list = self.inner.lock().map_err(|_| ConcurrentListError::LockAcquisitionError)?;
        Ok(list.get(index).cloned())
    }

    /// Returns the number of items in the list in a thread-safe manner.
    ///
    /// # Returns
    /// A `Result` containing the number of items, or an error if the lock failed.
    pub fn len(&self) -> Result<usize, ConcurrentListError> {
        let list = self.inner.lock().map_err(|_| ConcurrentListError::LockAcquisitionError)?;
        Ok(list.len())
    }

    /// Returns `true` if the list contains no elements.
    ///
    /// # Returns
    /// A `Result` containing a boolean value, or an error if the lock failed.
    pub fn is_empty(&self) -> Result<bool, ConcurrentListError> {
        let list = self.inner.lock().map_err(|_| ConcurrentListError::LockAcquisitionError)?;
        Ok(list.is_empty())
    }
}