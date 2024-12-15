//! Defines the Scalar trait and its implementations.
//!
//! The Scalar trait represents numeric types that can be used in vectors and matrices.
//! It requires:
//! - Basic arithmetic operations (+, -, *, /)
//! - Identity elements (zero and one)
//! - Formatting capabilities (Debug and Display)

use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

/// A type that can be used as a scalar in linear algebra operations.
///
/// The zero() and one() methods are required because:
/// - zero(): Needed for vector/matrix initialization and addition identity
/// - one(): Needed for matrix diagonal initialization and multiplication identity
///
/// # Example
/// ```
/// use matrix::Scalar;
///
/// // Using built-in f32 implementation
/// let a: f32 = 2.0;
/// let b: f32 = 3.0;
/// let sum = a + b;
///
/// // Creating a zero vector needs Scalar::zero()
/// let zero = f32::zero();
/// ```
pub trait Scalar:
    Copy +                    // Values must be copyable
    Add<Output = Self> +      // a + b
    Sub<Output = Self> +      // a - b
    Mul<Output = Self> +      // a * b
    Div<Output = Self> +      // a / b
    Debug +                   // println!("{:?}", a)
    Display                   // println!("{}", a)
{
    /// Returns the additive identity (zero) for this type.
    /// This element satisfies a + zero() = a for all a.
    fn zero() -> Self;

    /// Returns the multiplicative identity (one) for this type.
    /// This element satisfies a * one() = a for all a.
    fn one() -> Self;
}

/// Implementation of Scalar trait for f32.
impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_scalar_zero() {
        assert_eq!(f32::zero(), 0.0);
    }

    #[test]
    fn test_f32_scalar_one() {
        assert_eq!(f32::one(), 1.0);
    }

    #[test]
    fn test_f32_scalar_operations() {
        let a: f32 = 2.0;
        let b: f32 = 3.0;

        assert_eq!(a + b, 5.0);
        assert_eq!(a - b, -1.0);
        assert_eq!(a * b, 6.0);
        assert_eq!(a / b, 2.0 / 3.0);
    }
}
