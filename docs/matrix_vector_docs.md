# FRD-PU Matrix Vector Module Documentation

*Because who needs BLAS when you can roll your own matrix multiplication? 🤡*

## Overview

This module is the linear algebra equivalent of "I can fix her" energy - it's a zero-dependency, high-performance matrix and vector library that thinks it can replace decades of optimized math libraries with some spicy SIMD instructions. Spoiler alert: it's actually pretty decent! 

The module provides two APIs:
1. **Default Memory Management**: For when you want Rust to hold your hand like a caring parent
2. **Manual Memory Management**: For when you're feeling dangerous and want to manage your own memory like it's 1995

## Core Types

### `Matrix<'a>`

A matrix that stores `f64` values in row-major order (because we're not monsters who use column-major).

```rust
pub struct Matrix<'a> {
    data: Vec<f64>,     // The actual numbers, stored flat like your jokes
    rows: usize,        // How many rows we pretending to have
    cols: usize,        // How many columns we got
    _phantom: std::marker::PhantomData<&'a mut [f64]>, // Lifetime wizardry ✨
}
```

#### Constructors

##### `Matrix::new(rows: usize, cols: usize) -> Self`
Creates a new matrix filled with zeros, because starting from nothing is very relatable.

**Example:**
```rust
let matrix = Matrix::new(3, 4); // 3x4 matrix of pure disappointment (zeros)
```

##### `Matrix::from_slice_mut<'a>(rows: usize, cols: usize, data_slice: &'a mut [f64]) -> Matrix<'a>`
The "I brought my own data" constructor. Takes a mutable slice and yeets it into a Vec using some `unsafe` magic that would make your CS professor cry.

**⚠️ WARNING:** This function uses `Vec::from_raw_parts` and `std::mem::forget`. It's like doing parkour - looks cool but one wrong move and you're debugging memory corruption at 3 AM.

**Panics:** If your slice size doesn't match `rows * cols`. The code will roast you harder than Gordon Ramsay.

**Example:**
```rust
let mut data = vec![1.0, 2.0, 3.0, 4.0];
let matrix = Matrix::from_slice_mut(2, 2, &mut data);
```

#### Methods

##### `get(&self, row: usize, col: usize) -> Option<&f64>`
Gets an element without panicking like a civilized function. Returns `None` if you're trying to access the matrix equivalent of your dating life (out of bounds).

##### `get_mut(&mut self, row: usize, col: usize) -> Option<&mut f64>`
Like `get()` but spicier - lets you actually change the values.

##### `mul_matrix_fast(&self, other: &Self) -> Option<Self>`
The star of the show! Matrix multiplication that's faster than your ex's rebound relationship.

**How it works:**
- Checks if dimensions are compatible (unlike your relationship choices)
- Automatically picks between standard and SIMD implementations
- Returns `None` if dimensions don't match (with a passive-aggressive error message)

**SIMD Magic:** If your CPU supports AVX2, it processes 4 elements at once like a mathematical speed demon.

### `Vector<'a>`

A vector that's basically a fancy array with delusions of grandeur.

```rust
pub struct Vector<'a> {
    data: Vec<f64>,     // The numbers go brrr
    len: usize,         // How long is this thing
    _phantom: std::marker::PhantomData<&'a mut [f64]>, // More lifetime sorcery
}
```

#### Constructors

##### `Vector::new(len: usize) -> Self`
Creates a zero-filled vector for when you need to start your disappointment from scratch.

##### `Vector::from_slice_mut<'a>(len: usize, data_slice: &'a mut [f64]) -> Vector<'a>`
Same energy as the Matrix version - takes your slice and does unsafe things to it.

#### Methods

##### `get(&self, index: usize) -> Option<&f64>`
Safe element access that won't crash your program (unlike your life choices).

##### `get_mut(&mut self, index: usize) -> Option<&mut f64>`
Mutable access for when you need to change things up.

##### `dot(&self, other: &Self) -> Option<f64>`
Calculates the dot product with SIMD optimization because regular multiplication is for peasants.

Returns `None` if vector lengths don't match (skill issue).

## SIMD Optimization

This library is flexing harder than a gym bro with its AVX2 optimizations:

### Matrix Multiplication (`mul_matrix_avx`)
- Processes 4 `f64` elements simultaneously
- Uses AVX2 instructions like `_mm256_add_pd` and `_mm256_mul_pd`
- Handles remainder elements that don't fit in 4-element chunks
- Wrapped in `unsafe` because we live dangerously

### Vector Dot Product
- Also processes 4 elements at a time
- Falls back to standard implementation on non-AVX2 CPUs
- Because nobody likes being left out

## Memory Management APIs

### Default API (Recommended for Beginners)
Uses standard Rust `Vec` and global allocator. Safe, predictable, boring.

```rust
let mut matrix = Matrix::new(100, 100);
let mut vector = Vector::new(100);
// Rust handles memory like a responsible adult
```

### Manual API (For the Brave/Foolish)
Takes pre-allocated slices and does dark magic with `Vec::from_raw_parts`.

```rust
let mut data = vec![0.0; 10000];
let matrix = Matrix::from_slice_mut(100, 100, &mut data);
// You're now responsible for not breaking everything
```

## Compile-Time Features

The library uses conditional compilation for SIMD:
- `#[cfg(target_feature = "avx2")]` - For the chosen CPUs
- `#[cfg(not(target_feature = "avx2"))]` - For everyone else (sad)

## Error Handling Philosophy

This library believes in:
- Returning `Option` types instead of panicking (mostly)
- Helpful error messages via `eprintln!`
- Assertions that will absolutely destroy you if you mess up dimensions

## Performance Characteristics

- **Matrix Multiplication**: O(n³) but with SIMD seasoning
- **Vector Dot Product**: O(n) with 4x parallelization where possible
- **Memory Usage**: Row-major storage, cache-friendly access patterns

## Usage Examples

```rust
// Create matrices the easy way
let mut a = Matrix::new(3, 3);
let mut b = Matrix::new(3, 3);

// Fill them with actual data (not shown because that's homework)

// Multiply like a boss
if let Some(result) = a.mul_matrix_fast(&b) {
    println!("Success! We did math!");
} else {
    println!("Skill issue detected");
}

// Vector operations
let v1 = Vector::new(1000);
let v2 = Vector::new(1000);

if let Some(dot_product) = v1.dot(&v2) {
    println!("Dot product: {}", dot_product);
}
```

## Future Improvements (AKA TODO List)

According to the comments, future work includes:
- Matrix inversion (good luck with numerical stability)
- Determinant calculation (prepare for floating-point tears)
- LU decomposition (because we clearly need more complexity)
- SVD (at this point, just use a real library)

## Safety Considerations

The manual memory management API is about as safe as juggling chainsaws:
- Uses `unsafe` blocks with `Vec::from_raw_parts`
- Caller must ensure proper lifetime management
- One mistake and you're debugging memory corruption
- The code comments are more optimistic about safety than a startup's burn rate

## Platform Support

- **x86-64 with AVX2**: Full SIMD optimization
- **Everything else**: Falls back to standard implementations
- **Your sanity**: Not guaranteed on any platform

## Final Verdict

This library is like that friend who insists on making everything from scratch - admirable dedication, questionable life choices, but surprisingly good results. It's perfect for when you want to feel superior to people using "bloated" libraries like NumPy or Eigen, while secretly hoping nobody benchmarks your code against actual optimized BLAS implementations.

*Remember: With great SIMD comes great segfaults.* 🚀💥