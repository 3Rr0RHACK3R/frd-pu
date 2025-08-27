// src/matrix_vector.rs

//! # High-Performance Linear Algebra Module (Zero-Dependency)
//!
//! This module provides foundational data structures and operations for linear algebra,
//! including matrices and vectors, without relying on any external or internal dependencies.
//! It is designed for maximum performance by leveraging CPU-specific instructions (SIMD)
//! when available.
//!
//! This module offers two distinct APIs for memory management:
//!
//! 1.  **Default Memory Management:** Uses standard Rust `Vec` and the global allocator,
//!     providing a convenient and safe API for general use.
//! 2.  **Slice-based Initialization:** This API provides constructors that take a mutable
//!     slice (`&'a mut [f64]`) and create a new instance by **copying** the data.
//!     This allows the user to initialize a `Matrix` or `Vector` from a pre-existing
//!     memory region, like a memory pool, while ensuring safe memory management within the struct.
//!
//! Future work includes adding more advanced operations such as matrix inversion,
//! determinant calculation, and decomposition algorithms like LU or SVD.

// Conditionally enable SIMD support for specific CPU architectures.
// This example uses AVX2 for `f64` types, common on modern x86-64 CPUs.
#[cfg(target_feature = "avx2")]
use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_setzero_pd, _mm256_storeu_pd};

/// A fixed-size matrix with elements of type `f64`.
/// The data is stored in a flat vector in row-major order.
#[derive(Debug, Clone)]
pub struct Matrix<'a> {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
    // A lifetime `'a` is used to indicate that the struct's creation
    // might be tied to the lifetime of an input slice, even though it owns its data.
    _phantom: std::marker::PhantomData<&'a mut [f64]>,
}

impl Matrix<'_> {
    /// Creates a new `Matrix` with default memory allocation using `Vec`.
    ///
    /// # Arguments
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new `Matrix` by copying data from a pre-allocated mutable slice of `f64`.
    ///
    /// The data from the slice is cloned into a new `Vec`, ensuring the `Matrix`
    /// owns its data and manages its own memory safely.
    ///
    /// # Arguments
    /// * `rows` - The number of rows.
    /// * `cols` - The number of columns.
    /// * `data_slice` - A mutable slice of `f64` whose data will be copied.
    ///
    /// # Panics
    /// Panics if the size of the slice does not match `rows * cols`.
    pub fn from_slice_mut<'a>(rows: usize, cols: usize, data_slice: &'a mut [f64]) -> Matrix<'a> {
        assert_eq!(data_slice.len(), rows * cols, "Slice size must match matrix dimensions.");
        
        Matrix {
            data: data_slice.to_vec(),
            rows,
            cols,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Gets an immutable reference to an element at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> Option<&f64> {
        if row < self.rows && col < self.cols {
            self.data.get(row * self.cols + col)
        } else {
            None
        }
    }

    /// Gets a mutable reference to an element at `(row, col)`.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64> {
        if row < self.rows && col < self.cols {
            self.data.get_mut(row * self.cols + col)
        } else {
            None
        }
    }

    /// Multiplies this matrix by another matrix using the fastest available method.
    /// This function automatically selects between a standard implementation and
    /// a SIMD-optimized version if the CPU supports AVX2.
    ///
    /// This uses the default memory management for the output matrix.
    pub fn mul_matrix_fast(&self, other: &Self) -> Option<Self> {
        if self.cols != other.rows {
            eprintln!("Error: Mismatched dimensions for matrix multiplication");
            return None;
        }

        let mut result = Self::new(self.rows, other.cols);

        // Check if AVX2 is available.
        #[cfg(target_feature = "avx2")]
        {
            // Use the SIMD-optimized version.
            Self::mul_matrix_avx(self, other, &mut result);
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            // Fallback to the standard, non-SIMD version.
            Self::mul_matrix_standard(self, other, &mut result);
        }

        Some(result)
    }

    /// Standard, non-SIMD matrix multiplication implementation.
    /// This is the fallback if no special CPU features are available.
    fn mul_matrix_standard(a: &Self, b: &Self, result: &mut Self) {
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum = 0.0;
                for k in 0..a.cols {
                    sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
                }
                result.data[i * b.cols + j] = sum;
            }
        }
    }
    
    /// Internal helper to transpose a matrix, making it suitable for SIMD operations.
    fn transpose(m: &Self) -> Self {
        let mut transposed = Self::new(m.cols, m.rows);
        for i in 0..m.rows {
            for j in 0..m.cols {
                transposed.data[j * m.rows + i] = m.data[i * m.cols + j];
            }
        }
        transposed
    }
    
    #[cfg(target_feature = "avx2")]
    /// A SIMD-optimized matrix multiplication using AVX2 instructions.
    /// This method loads and processes 4 elements at a time.
    /// For efficient SIMD access, the second matrix `b` is transposed internally.
    // Safety: This function requires the `avx2` target feature to be enabled.
    fn mul_matrix_avx(a: &Self, b: &Self, result: &mut Self) {
        let b_transposed = Self::transpose(b);
        unsafe {
            for i in 0..a.rows {
                for j in 0..b_transposed.rows { // now j corresponds to the original b.cols
                    let mut sum_vec = _mm256_setzero_pd();
                    let mut k = 0;
                    
                    // Process 4 elements at a time
                    while k + 4 <= a.cols {
                        let a_vec = _mm256_loadu_pd(&a.data[i * a.cols + k]);
                        // Accessing transposed B is now row-major and thus SIMD-friendly
                        let b_vec = _mm256_loadu_pd(&b_transposed.data[j * b_transposed.cols + k]);
                        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
                        k += 4;
                    }

                    // Horizontal sum of the accumulator vector
                    let mut sum_arr = [0.0f64; 4];
                    _mm256_storeu_pd(sum_arr.as_mut_ptr(), sum_vec);
                    let mut sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

                    // Handle any remaining elements that don't fit into a 4-element block.
                    while k < a.cols {
                         sum += a.data[i * a.cols + k] * b_transposed.data[j * b_transposed.cols + k];
                         k += 1;
                    }
                    result.data[i * result.cols + j] = sum;
                }
            }
        }
    }
}

/// A fixed-size vector with elements of type `f64`.
#[derive(Debug, Clone)]
pub struct Vector<'a> {
    data: Vec<f64>,
    len: usize,
    _phantom: std::marker::PhantomData<&'a mut [f64]>,
}

impl Vector<'_> {
    /// Creates a new `Vector` with default memory allocation.
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
            len,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new `Vector` by copying data from a pre-allocated mutable slice of `f64`.
    ///
    /// The data from the slice is cloned into a new `Vec`, ensuring the `Vector`
    /// owns its data and manages its own memory safely.
    ///
    /// # Panics
    /// Panics if the size of the slice does not match `len`.
    pub fn from_slice_mut<'a>(len: usize, data_slice: &'a mut [f64]) -> Vector<'a> {
        assert_eq!(data_slice.len(), len, "Slice size must match vector length.");
        
        Vector {
            data: data_slice.to_vec(),
            len,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Gets an immutable reference to an element at `index`.
    pub fn get(&self, index: usize) -> Option<&f64> {
        self.data.get(index)
    }

    /// Gets a mutable reference to an element at `index`.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut f64> {
        self.data.get_mut(index)
    }

    /// Calculates the dot product of two vectors.
    pub fn dot(&self, other: &Self) -> Option<f64> {
        if self.len != other.len {
            return None;
        }

        let mut sum;
        
        #[cfg(target_feature = "avx2")]
        {
            // Use SIMD-optimized dot product.
            unsafe {
                let mut sum_vec = _mm256_setzero_pd();
                let mut i = 0;
                while i + 4 <= self.len {
                    let a_vec = _mm256_loadu_pd(&self.data[i]);
                    let b_vec = _mm256_loadu_pd(&other.data[i]);
                    sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec));
                    i += 4;
                }

                let mut sum_arr: [f64; 4] = [0.0; 4];
                _mm256_storeu_pd(sum_arr.as_mut_ptr(), sum_vec);
                sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
                
                while i < self.len {
                    sum += self.data[i] * other.data[i];
                    i += 1;
                }
            }
        }
        
        #[cfg(not(target_feature = "avx2"))]
        {
            // Fallback to standard dot product.
            sum = 0.0;
            for i in 0..self.len {
                sum += self.data[i] * other.data[i];
            }
        }
        
        Some(sum)
    }
}