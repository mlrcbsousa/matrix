#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]

//! Linear algebra library implementing basic vector and matrix operations.
//!
//! This library provides implementations for vector and matrix operations
//! including addition, multiplication, determinants, and more.

mod complex;
mod lerp;
mod matrix;
mod projection;
mod scalar;
mod vector;

pub use complex::Complex;
pub use lerp::lerp;
pub use matrix::Matrix;
pub use projection::projection;
pub use scalar::Scalar;
pub use vector::{angle_cos, cross_product, linear_combination, Vector};
