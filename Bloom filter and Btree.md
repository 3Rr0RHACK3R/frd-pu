FRD-PU: Version 2.0 - Data Structures
This document provides an overview of the new, highly-optimized data structures introduced in Version 2.0 of the FRD-PU library. These additions, built with our core philosophy of efficiency and zero dependencies, are designed to solve complex computational problems with minimal memory and processing overhead.

BloomFilter
The BloomFilter module introduces a space-efficient, probabilistic data structure used to test whether an element is a member of a set. It offers a significant memory advantage over traditional hash sets but comes with the trade-off of a small chance of false positives. It guarantees no false negatives.

Key Features
Memory Efficiency: Drastically reduces memory footprint for set membership tests.

Fast Operations: Provides extremely fast add and check operations.

Zero Dependencies: Relies only on the Rust standard library.

Example Usage
The following example demonstrates how to create a Bloom filter and perform basic operations.

use frd_pu::bloom_filter::{BloomFilter, BloomFilterError};

fn main() -> Result<(), BloomFilterError> {
    // Create a new Bloom filter with a capacity of 1000 items and a 1% false positive probability.
    let mut filter = BloomFilter::new(1000, 0.01)?;

    // Add items to the filter.
    filter.add(&"professional");
    filter.add(&"project");
    filter.add(&"efficiency");

    // Check for membership.
    assert_eq!(filter.check(&"project"), true);
    assert_eq!(filter.check(&"quality"), false); // May be true in rare cases due to false positives.
    
    Ok(())
}

BinarySearchTree
The BinarySearchTree module provides a professional-grade, in-memory data structure for storing and retrieving key-value pairs. It is designed for operations that require fast lookups, insertions, and deletions, maintaining a sorted structure for efficient searching.

Key Features
Logarithmic Complexity: Provides O(
logn) average time complexity for core operations.

Zero Dependencies: Implemented with only the Rust standard library.

Generic: Supports any key-value pair that implements the Ord trait for comparison.

Example Usage
This example shows how to insert and search for elements within the Binary Search Tree.

use frd_pu::btree::{BinarySearchTree, BinarySearchTreeError};

fn main() -> Result<(), BinarySearchTreeError> {
    // Create a new Binary Search Tree.
    let mut bst = BinarySearchTree::new();

    // Insert key-value pairs.
    bst.insert(5, "Task B")?;
    bst.insert(3, "Task A")?;
    bst.insert(8, "Task C")?;

    // Search for a value.
    assert_eq!(bst.search(&3), Some(&"Task A"));
    assert_eq!(bst.search(&10), None);

    Ok(())
}
