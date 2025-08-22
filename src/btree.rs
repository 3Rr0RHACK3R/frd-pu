// src/btree.rs

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// Error type for B-Tree operations.
#[derive(Debug, PartialEq)]
pub enum BTreeError {
    /// Indicates that a key already exists in the tree.
    KeyAlreadyExists,
}

impl fmt::Display for BTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BTreeError::KeyAlreadyExists => write!(f, "The key already exists in the tree."),
        }
    }
}

impl Error for BTreeError {}

/// Represents a node in the Binary Search Tree.
/// This structure provides the core logic for a balanced tree like a B-Tree,
/// allowing for efficient ordered storage without external dependencies.
#[derive(Debug)]
struct Node<K: Ord, V> {
    key: K,
    value: V,
    left: Option<Box<Node<K, V>>>,
    right: Option<Box<Node<K, V>>>,
}

impl<K: Ord, V> Node<K, V> {
    /// Creates a new node with the given key and value.
    fn new(key: K, value: V) -> Self {
        Node {
            key,
            value,
            left: None,
            right: None,
        }
    }
}

/// A professional-grade, zero-dependency Binary Search Tree.
///
/// This data structure is a foundational concept for more complex trees like B-Trees.
/// It provides high-performance ordered storage and retrieval of key-value pairs,
/// adhering to the FRD-PU philosophy of being memory-first and dependency-free.
///
/// # Examples
///
/// ```
/// use frd_pu::btree::BTree;
///
/// let mut tree = BTree::new();
///
/// // Insert key-value pairs.
/// tree.insert("apple", 1).unwrap();
/// tree.insert("banana", 2).unwrap();
/// tree.insert("cherry", 3).unwrap();
///
/// // Retrieve values.
/// assert_eq!(tree.search("banana"), Some(&2));
/// assert_eq!(tree.search("grape"), None);
/// ```
#[derive(Debug, Default)]
pub struct BTree<K: Ord, V> {
    root: Option<Box<Node<K, V>>>,
}

impl<K: Ord, V> BTree<K, V> {
    /// Creates a new, empty B-Tree.
    pub fn new() -> Self {
        BTree { root: None }
    }

    /// Inserts a new key-value pair into the tree.
    ///
    /// # Arguments
    /// * `key` - The key to insert.
    /// * `value` - The value associated with the key.
    ///
    /// # Returns
    /// `Ok(())` if the insertion was successful, or a `BTreeError` if the key already exists.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), BTreeError> {
        let new_node = Box::new(Node::new(key, value));
        if self.root.is_none() {
            self.root = Some(new_node);
            return Ok(());
        }

        let mut current = self.root.as_mut().unwrap();
        loop {
            match new_node.key.cmp(&current.key) {
                Ordering::Less => {
                    if let Some(left) = current.left.as_mut() {
                        current = left;
                    } else {
                        current.left = Some(new_node);
                        return Ok(());
                    }
                }
                Ordering::Greater => {
                    if let Some(right) = current.right.as_mut() {
                        current = right;
                    } else {
                        current.right = Some(new_node);
                        return Ok(());
                    }
                }
                Ordering::Equal => {
                    return Err(BTreeError::KeyAlreadyExists);
                }
            }
        }
    }

    /// Searches for a key in the tree and returns a reference to its value.
    ///
    /// # Arguments
    /// * `key` - The key to search for.
    ///
    /// # Returns
    /// `Some(&V)` if the key is found, otherwise `None`.
    pub fn search(&self, key: &K) -> Option<&V> {
        let mut current = self.root.as_ref();
        while let Some(node) = current {
            match key.cmp(&node.key) {
                Ordering::Less => current = node.left.as_ref(),
                Ordering::Greater => current = node.right.as_ref(),
                Ordering::Equal => return Some(&node.value),
            }
        }
        None
    }
}
