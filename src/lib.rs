#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]

//! Linear algebra library implementing basic vector and matrix operations.
//!
//! This library provides implementations for vector and matrix operations
//! including addition, multiplication, determinants, and more.
//!
//! The scalar type is implemented as a trait, allowing for custom scalar types
//! to be used in vector and matrix operations. In this library, the scalar type
//! is implemented for `f32`, `i32` and `Complex` (which has its internal
//! components as `f32`). In a fully featured linear algebra library, the scalar
//! type would be implemented for more types, such as `f64`, `i64`, and others.
//!
//! > **Disclaimer**: This library is a learning project and is not intended for production use.
//! > It is part of the 42 school curriculum and is meant to be used for educational purposes.
//! > The library is not optimized for performance and may contain bugs or incorrect implementations.
//! > Use at your own risk.
//!
//! Panics are used more then they "should" be, the context for this is that the library was
//! a first exposure to Rust for me. For sure one improvement would be to use `Result` instead
//! of `panic!` in most cases. And have several nice Error types. Or also use `Option` more.
//! Also it wasn't clear from the project subject that I could use `Result` or `Option` instead
//! of `panic!`, given the method signatures were already defined.
//!
//! Also, probably the internal `data` of `Matrix` and `Vector` should be private and have
//! getters and setters, but it was convenient to have it public, given it was not a
//! requirement of the project.
//!
//! Finally, there are more unit tests than is usual in a library, or even reasonable, but
//! it was to make sure I would pass the evaluation of the project.
//!
//! If you find any issues or have suggestions for improvements, please open an issue on the
//! [GitHub repository](https://github.com/mlrcbsousa/matrix).

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
