// src/quicksort.rs

use std::error::Error;
use std::fmt;
use std::mem;

/// Error type for QuickSort operations.
#[derive(Debug, PartialEq)]
pub enum QuickSortError {
    /// Indicates that the provided slice is empty and cannot be sorted.
    EmptySlice,
}

impl fmt::Display for QuickSortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuickSortError::EmptySlice => write!(f, "The slice provided is empty and cannot be sorted."),
        }
    }
}

impl Error for QuickSortError {}

/// The public-facing QuickSort function.
///
/// This function sorts a mutable slice in place using the QuickSort algorithm.
/// It is generic over any type `T` that implements `PartialOrd` for comparison.
///
/// # Arguments
/// * `slice` - A mutable slice of data to be sorted.
///
/// # Returns
/// A `Result` that is `Ok(())` on success or `Err(QuickSortError)` if the slice is empty.
///
/// # Examples
///
/// ```
/// use frd_pu::quicksort::quicksort;
///
/// let mut data = vec![5, 2, 8, 1, 9];
/// quicksort(&mut data).unwrap();
/// assert_eq!(data, vec![1, 2, 5, 8, 9]);
/// ```
pub fn quicksort<T: PartialOrd>(slice: &mut [T]) -> Result<(), QuickSortError> {
    if slice.is_empty() {
        return Err(QuickSortError::EmptySlice);
    }
    // Call the internal, recursive function.
    quicksort_recursive(slice, 0, (slice.len() - 1) as isize);
    Ok(())
}

/// The main recursive function for QuickSort.
///
/// This function is where the core logic of the QuickSort algorithm resides. It
/// partitions the slice and recursively calls itself on the sub-slices.
///
/// # Arguments
/// * `slice` - The mutable slice of data to be sorted.
/// * `low` - The lower index of the current sub-slice.
/// * `high` - The upper index of the current sub-slice.
fn quicksort_recursive<T: PartialOrd>(slice: &mut [T], low: isize, high: isize) {
    // If the low index is less than the high index, there's at least one element to sort.
    if low < high {
        // Find the pivot and partition the array.
        let pivot_index = partition(slice, low, high);

        // Recursively sort the sub-array before the pivot.
        quicksort_recursive(slice, low, pivot_index - 1);

        // Recursively sort the sub-array after the pivot.
        quicksort_recursive(slice, pivot_index + 1, high);
    }
}

/// Partitions the slice around a pivot element.
///
/// This function uses the Lomuto partition scheme. It selects the last element
/// as the pivot and rearranges the slice so that all elements smaller than
/// the pivot come before it, and all elements larger than it come after.
///
/// # Arguments
/// * `slice` - The mutable slice to be partitioned.
/// * `low` - The starting index of the partition.
/// * `high` - The ending index of the partition.
///
/// # Returns
/// The final index of the pivot element after partitioning.
fn partition<T: PartialOrd>(slice: &mut [T], low: isize, high: isize) -> isize {
    let pivot_index = high as usize;
    let mut i = low - 1;

    for j in low..high {
        // Check if the current element is less than or equal to the pivot.
        if slice[j as usize] <= slice[pivot_index] {
            // Increment the index of the smaller element.
            i += 1;
            // Swap the elements at `i` and `j`.
            slice.swap(i as usize, j as usize);
        }
    }

    // Swap the pivot element with the element at `i + 1`.
    slice.swap((i + 1) as usize, pivot_index);

    // Return the final position of the pivot.
    i + 1
}

