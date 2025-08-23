// src/quicksort.rs

use std::error::Error;
use std::fmt;

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
    quicksort_recursive(slice, 0, (slice.len() - 1) as isize);
    Ok(())
}

/// The recursive core of the QuickSort algorithm.
///
/// # Arguments
/// * `slice` - The mutable slice to be sorted.
/// * `low` - The starting index of the sub-array.
/// * `high` - The ending index of the sub-array.
fn quicksort_recursive<T: PartialOrd>(slice: &mut [T], low: isize, high: isize) {
    if low < high {
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

    // Move the pivot to its final sorted position.
    slice.swap((i + 1) as usize, pivot_index);
    i + 1
}