// src/btree.rs

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// Error type for Binary Search Tree operations.
#[derive(Debug, PartialEq)]
pub enum BinarySearchTreeError {
    /// Indicates that a key already exists in the tree.
    KeyAlreadyExists,
}

impl fmt::Display for BinarySearchTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinarySearchTreeError::KeyAlreadyExists => write!(f, "The key already exists in the tree."),
        }
    }
}

impl Error for BinarySearchTreeError {}

/// Represents a node in the Binary Search Tree.
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

/// A professional-grade, zero-dependency Binary Search Tree (BST) data structure.
///
/// **Note:** This is a simple binary search tree and is not self-balancing. Performance
/// may degrade to O(n) in worst-case scenarios with already sorted data.
///
/// # Panics
///
/// This implementation will panic if it encounters a key already in the tree.
/// This is not ideal, and a more robust implementation would return a `Result`.
#[derive(Debug)]
pub struct BinarySearchTree<K: Ord, V> {
    root: Option<Box<Node<K, V>>>,
}

impl<K: Ord + fmt::Debug, V> BinarySearchTree<K, V> {
    /// Creates a new empty Binary Search Tree.
    pub fn new() -> Self {
        BinarySearchTree { root: None }
    }

    /// Inserts a key-value pair into the tree.
    ///
    /// # Arguments
    /// * `key` - The key to insert.
    /// * `value` - The value associated with the key.
    ///
    /// # Returns
    /// `Ok(())` on success, or a `BinarySearchTreeError` if the key already exists.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), BinarySearchTreeError> {
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
                    return Err(BinarySearchTreeError::KeyAlreadyExists);
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