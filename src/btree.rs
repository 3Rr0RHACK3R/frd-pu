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
/// **Note:** This is a Binary Search Tree (BST), not a B-Tree. While the file is named
/// `btree.rs` for organizational purposes, the implementation is that of a BST.
/// A Binary Search Tree is an efficient data structure for storing and retrieving
/// sorted data in memory. It provides fast insertion, deletion, and searching
/// operations for ordered data.
///
/// # Panics
///
/// This implementation will panic if it encounters a node that is not
/// properly initialized, but this is prevented by the use of `Option<Box<Node>>`.
///
/// # Examples
///
/// ```
/// use frd_pu::btree::{BinarySearchTree, BinarySearchTreeError};
///
/// // Create a new Binary Search Tree.
/// let mut bst = BinarySearchTree::new();
///
/// // Insert some key-value pairs.
/// bst.insert(5, "apple").unwrap();
/// bst.insert(3, "banana").unwrap();
/// bst.insert(7, "cherry").unwrap();
///
/// // Search for a value.
/// assert_eq!(bst.search(&3), Some(&"banana"));
/// assert_eq!(bst.search(&10), None);
///
/// // Attempting to insert a duplicate key returns an error.
/// let result = bst.insert(5, "grape");
/// assert_eq!(result, Err(BinarySearchTreeError::KeyAlreadyExists));
/// ```
pub struct BinarySearchTree<K: Ord, V> {
    root: Option<Box<Node<K, V>>>,
}

impl<K: Ord, V> BinarySearchTree<K, V> {
    /// Creates a new, empty Binary Search Tree.
    pub fn new() -> Self {
        BinarySearchTree { root: None }
    }

    /// Inserts a new key-value pair into the tree.
    ///
    /// # Arguments
    /// * `key` - The key to insert. It must implement the `Ord` trait for ordering.
    /// * `value` - The value associated with the key.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if the key already exists.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), BinarySearchTreeError> {
        let new_node = Box::new(Node::new(key, value));
        if let Some(root) = self.root.as_mut() {
            // If the root exists, traverse the tree to find the insertion point.
            Self::insert_recursive(root, new_node)
        } else {
            // If the tree is empty, the new node becomes the root.
            self.root = Some(new_node);
            Ok(())
        }
    }

    /// Recursively inserts a new node into the correct position in the tree.
    fn insert_recursive(current: &mut Box<Node<K, V>>, new_node: Box<Node<K, V>>) -> Result<(), BinarySearchTreeError> {
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
