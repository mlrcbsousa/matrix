#![warn(missing_docs)]

//! Linear algebra library implementing basic vector and matrix operations.
//!
//! This library provides implementations for vector and matrix operations
//! including addition, multiplication, determinants, and more.

mod matrix;
mod scalar;
mod vector;

pub use matrix::Matrix;
pub use scalar::Scalar;
pub use vector::linear_combination;
pub use vector::Vector;
