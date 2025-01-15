//! Defines the Scalar trait and its implementations.
//!
//! The Scalar trait represents numeric types that can be used in vectors and matrices.
//! It requires:
//! - Basic arithmetic operations (+, -, *, /)
//! - Identity elements (zero and one)
//! - Formatting capabilities (Debug and Display)

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A type that can be used as a scalar in linear algebra operations.
///
/// The `zero()` and `one()` methods are required because:
/// - `zero()`: Needed for vector/matrix initialization and addition identity
/// - `one()`: Needed for matrix diagonal initialization and multiplication identity
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
    Copy +                  // Values must be copyable
    Add<Output = Self> +    // a + b
    Sub<Output = Self> +    // a - b
    Mul<Output = Self> +    // a * b
    Div<Output = Self> +    // a / b
    AddAssign +             // a += b
    SubAssign +             // a -= b
    MulAssign +             // a *= b
    DivAssign +             // a /= b
    Debug +                 // println!("{:?}", a)
    Display +               // println!("{}", a)
    Neg<Output = Self> +    // -a
    PartialEq +             // a == b
    PartialOrd              // a < b
{
    /// Returns the additive identity (zero) for this type.
    /// This element satisfies `a + zero() = a` for all `a`.
    fn zero() -> Self;

    /// Returns the multiplicative identity (one) for this type.
    /// This element satisfies `a * one() = a` for all `a`.
    fn one() -> Self;

    /// Convert to f32.
    /// This is useful for converting scalar types to f32 for operations like dot products.
    /// Not using `Into<f32>` to make it explicit that the conversion is happening, as well as
    /// because it wouldn't work to implement `Into<f32>` for any primitive K (orphan rule).
    fn to_f32(&self) -> f32;

    /// Performs fused multiply-add: (a * b) + c
    ///
    /// This operation is performed in a single step with a single rounding,
    /// providing better precision than separate multiplication and addition.
    /// The default implementation multiplies then adds, but specialized
    /// implementations (like `f32`) use hardware FMA instructions.
    ///
    /// # Hardware Implementation
    /// On x86_64 with FMA support, this maps to SIMD instructions:
    /// - **vfmadd132ps**: (a * b) + c  where a is the first source operand
    /// - **vfmadd213ps**: (a * b) + c  where b is the first source operand
    /// - **vfmadd231ps**: (a * b) + c  where c is the first source operand
    ///
    /// Rust provides safe access to these through `mul_add()` on floating point types,
    /// which internally may use the intrinsic `fmaf32`.
    ///
    /// # Performance
    /// Hardware implementations can be faster than separate multiply and add operations.
    ///
    /// # Precision
    /// FMA provides better precision by eliminating an intermediate rounding step:
    /// - Regular: (a * b) rounds to N bits, then + c rounds again to N bits
    /// - FMA: Computes (a * b) + c to full precision before single final rounding
    ///
    /// # Arguments
    /// * `a` - First multiplication operand
    /// * `b` - Second multiplication operand
    /// * `c` - Value to add to product
    ///
    /// # Example
    /// ```
    /// use matrix::Scalar;
    ///
    /// let result = f32::fma(2.0, 3.0, 4.0); // Computes (2.0 * 3.0) + 4.0 = 10.0
    /// assert_eq!(result, 10.0);
    /// ```
    fn fma(a: Self, b: Self, c: Self) -> Self {
        // Default implementation - multiply then add
        (a * b) + c
    }
}

/// Implementation of `Scalar` trait for `f32`.
impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn to_f32(&self) -> f32 {
        *self
    }

    // Safe, portable FMA using std lib
    // Internally optimizes to appropriate FMA instruction
    // stabilized version of fmaf32 intrinsic
    fn fma(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c)
    }
}

/// Implementation of `Scalar` trait for `i32`.
impl Scalar for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn to_f32(&self) -> f32 {
        *self as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_zero() {
        assert_eq!(f32::zero(), 0.0);
    }

    #[test]
    fn test_scalar_one() {
        assert_eq!(f32::one(), 1.0);
    }

    #[test]
    fn test_f32_scalar_operations() {
        let mut a: f32 = 2.0;
        let b: f32 = 3.0;

        assert_eq!(a + b, 5.0);
        assert_eq!(a - b, -1.0);
        assert_eq!(a * b, 6.0);
        assert_eq!(a / b, 2.0 / 3.0);

        a += b;
        assert_eq!(a, 5.0);
        a -= b;
        assert_eq!(a, 2.0);
        a *= b;
        assert_eq!(a, 6.0);
        a /= b;
        assert_eq!(a, 2.0);
    }

    #[test]
    fn test_f32_fma() {
        let a = 2.0f32;
        let b = 3.0f32;
        let c = 4.0f32;

        let result = f32::fma(a, b, c);
        let expected = (a * b) + c;

        assert_eq!(result, expected);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_f32_fma_precision() {
        // Example where FMA provides better precision
        let a = 0.1f32;
        let b = 0.2f32;
        let c = 0.3f32;

        let fma_result = f32::fma(a, b, c);
        let standard = (a * b) + c;

        // Results may differ slightly due to FMA's single rounding
        assert!((fma_result - standard).abs() < f32::EPSILON);
    }
}
