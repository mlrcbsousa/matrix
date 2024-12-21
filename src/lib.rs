#![warn(missing_docs)]

//! Linear algebra library implementing basic vector and matrix operations.
//!
//! This library provides implementations for vector and matrix operations
//! including addition, multiplication, determinants, and more.

mod lerp;
mod matrix;
mod scalar;
mod vector;

pub use lerp::lerp;
pub use matrix::Matrix;
pub use scalar::Scalar;
pub use vector::{angle_cos, linear_combination, Vector};
