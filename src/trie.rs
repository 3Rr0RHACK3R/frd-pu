// src/trie.rs

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Error type for Trie operations.
#[derive(Debug, PartialEq)]
pub enum TrieError {
    /// Indicates an invalid character was used for an operation.
    InvalidCharacter(String),
}

impl fmt::Display for TrieError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrieError::InvalidCharacter(msg) => write!(f, "Invalid character in key: {}", msg),
        }
    }
}

impl Error for TrieError {}

/// Represents a node in the Trie.
/// Each node can have children and a flag indicating if it's the end of a word.
#[derive(Debug, Default)]
struct TrieNode {
    /// A map of characters to their child nodes.
    children: HashMap<char, TrieNode>,
    /// A flag to mark the end of a complete word.
    is_end_of_word: bool,
}

/// A professional-grade, zero-dependency Trie (Prefix Tree) data structure.
///
/// A Trie is ideal for efficient retrieval of a key in a dataset of strings,
/// making it perfect for applications like autocompletion, spell-checking,
/// and IP routing.
///
/// # Examples
///
/// ```
/// use frd_pu::trie::Trie;
///
/// let mut trie = Trie::new();
///
/// // Insert some words.
/// trie.insert("apple").unwrap();
/// trie.insert("application").unwrap();
///
/// // Check for the existence of words and prefixes.
/// assert!(trie.search("apple"));
/// assert!(trie.starts_with("app"));
/// assert!(!trie.search("app"));
/// ```
#[derive(Debug, Default)]
pub struct Trie {
    root: TrieNode,
}

impl Trie {
    /// Creates a new, empty Trie.
    pub fn new() -> Self {
        Trie {
            root: TrieNode::default(),
        }
    }

    /// Inserts a word into the Trie.
    ///
    /// # Arguments
    /// * `word` - The string slice to insert.
    ///
    /// # Returns
    /// `Ok(())` if the word was inserted successfully, or a `TrieError` if
    /// the word contains non-alphanumeric characters.
    pub fn insert(&mut self, word: &str) -> Result<(), TrieError> {
        let mut node = &mut self.root;
        for c in word.chars() {
            // A simple check for valid characters. Extend this as needed.
            if !c.is_alphanumeric() {
                return Err(TrieError::InvalidCharacter(c.to_string()));
            }
            node = node.children.entry(c).or_insert_with(TrieNode::default);
        }
        node.is_end_of_word = true;
        Ok(())
    }

    /// Checks if a complete word exists in the Trie.
    ///
    /// # Arguments
    /// * `word` - The string slice to search for.
    ///
    /// # Returns
    /// `true` if the word is found, otherwise `false`.
    pub fn search(&self, word: &str) -> bool {
        let node = self.get_node(word);
        node.map_or(false, |n| n.is_end_of_word)
    }

    /// Checks if a prefix exists in the Trie.
    ///
    /// # Arguments
    /// * `prefix` - The string slice to search for.
    ///
    /// # Returns
    /// `true` if the prefix is found, otherwise `false`.
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.get_node(prefix).is_some()
    }

    /// A private helper function to traverse the Trie to the node
    /// corresponding to the given word or prefix.
    fn get_node(&self, word: &str) -> Option<&TrieNode> {
        let mut node = &self.root;
        for c in word.chars() {
            match node.children.get(&c) {
                Some(child_node) => node = child_node,
                None => return None,
            }
        }
        Some(node)
    }
}
