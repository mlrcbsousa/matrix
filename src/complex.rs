//! Complex number implementation for linear algebra operations.
//!
//! Provides a Complex type that satisfies the Scalar trait requirements.
//! Includes standard complex number operations and conversions.

use crate::Scalar;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Complex number representation using real and imaginary parts.
///
/// # Examples
/// ```
/// use matrix::Complex;
///
/// // Create a complex number
/// let z = Complex::new(3.0, 4.0);
///
/// // Calculate its modulus
/// let m = z.modulus();
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Complex {
    /// Real part of the complex number
    pub r: f32,
    /// Imaginary part of the complex number
    pub i: f32,
}

impl Complex {
    /// Creates a new complex number.
    ///
    /// # Examples
    /// ```
    /// use matrix::Complex;
    ///
    /// let z = Complex::new(1.0, 2.0);
    /// ```
    pub fn new(r: f32, i: f32) -> Self {
        Self { r, i }
    }

    /// Returns the modulus (magnitude) of the complex number.
    ///
    /// # Examples
    /// ```
    /// use matrix::Complex;
    ///
    /// let z = Complex::new(3.0, 4.0);
    /// let m = z.modulus();
    /// ```
    pub fn modulus(&self) -> f32 {
        (self.r * self.r + self.i * self.i).sqrt()
    }

    // Other methods that could be implemented:
    // - `arg`: argument (angle) in radians.
    // - `conj`: complex conjugate (a - bi).
}

// Display implementation for pretty printing
impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.i >= 0.0 {
            write!(f, "{} + {}i", self.r, self.i)
        } else {
            write!(f, "{} - {}i", self.r, -self.i)
        }
    }
}

// Scalar trait implementation
impl Scalar for Complex {
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    // Implement conversion from Complex to f32 (for norm calculations)
    fn to_f32(&self) -> f32 {
        self.modulus()
    }
}

// Arithmetic operations implementations
impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.r + rhs.r, self.i + rhs.i)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.r += rhs.r;
        self.i += rhs.i;
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.r - rhs.r, self.i - rhs.i)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.r -= rhs.r;
        self.i -= rhs.i;
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.r * rhs.r - self.i * rhs.i,
            self.r * rhs.i + self.i * rhs.r,
        )
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        let r = self.r * rhs.r - self.i * rhs.i;
        let i = self.r * rhs.i + self.i * rhs.r;
        self.r = r;
        self.i = i;
    }
}

impl MulAssign<f32> for Complex {
    fn mul_assign(&mut self, rhs: f32) {
        self.r *= rhs;
        self.i *= rhs;
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let denom = rhs.r * rhs.r + rhs.i * rhs.i;
        Self::new(
            (self.r * rhs.r + self.i * rhs.i) / denom,
            (self.i * rhs.r - self.r * rhs.i) / denom,
        )
    }
}

impl DivAssign for Complex {
    fn div_assign(&mut self, rhs: Self) {
        let denom = rhs.r * rhs.r + rhs.i * rhs.i;
        let r = (self.r * rhs.r + self.i * rhs.i) / denom;
        let i = (self.i * rhs.r - self.r * rhs.i) / denom;
        self.r = r;
        self.i = i;
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.r, -self.i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{angle_cos, cross_product, lerp, linear_combination, Matrix, Vector};

    #[test]
    fn test_complex_methods() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.modulus(), 5.0);
    }

    #[test]
    fn test_scalar_ops() {
        let mut z = Complex::new(2.0, 3.0);
        z *= 2.0;
        assert_eq!(z, Complex::new(4.0, 6.0));
    }

    #[test]
    fn test_complex_lerp() {
        let a = Complex::new(1.0, 1.0);
        let b = Complex::new(3.0, 3.0);

        // Test t = 0 returns first value
        let result = lerp(a, b, 0.0);
        assert_eq!(result, a);

        // Test t = 1 returns second value
        let result = lerp(a, b, 1.0);
        assert_eq!(result, b);

        // Test t = 0.5 returns midpoint
        let result = lerp(a, b, 0.5);
        assert_eq!(result, Complex::new(2.0, 2.0));
    }

    #[test]
    fn test_complex_vector_norms() {
        // Create vector [3 + 4i]
        let v = Vector::new(vec![Complex::new(3.0, 4.0)]);

        // 1-norm should be |3 + 4i| = 5
        assert_eq!(v.norm_1(), 5.0);

        // 2-norm (Euclidean) should be |3 + 4i| = 5
        assert_eq!(v.norm(), 5.0);

        // inf-norm should be |3 + 4i| = 5
        assert_eq!(v.norm_inf(), 5.0);

        // Test vector [1 + i, 2 + 2i]
        let v = Vector::new(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);

        // 1-norm: |1 + i| + |2 + 2i| = √2 + 2√2
        assert!((v.norm_1() - (2.0f32.sqrt() * 3.0)).abs() < 1e-6);

        // 2-norm: √(|1 + i|² + |2 + 2i|²) = √(2 + 8) = √10
        assert!((v.norm() - 10.0f32.sqrt()).abs() < 1e-6);

        // inf-norm: max(|1 + i|, |2 + 2i|) = 2√2
        assert!((v.norm_inf() - (2.0f32.sqrt() * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_complex_into_f32() {
        let z = Complex::new(3.0, 4.0);
        let magnitude: f32 = z.to_f32();
        assert_eq!(magnitude, 5.0);
    }

    #[test]
    fn test_complex_display() {
        let z1 = Complex::new(1.0, 2.0);
        assert_eq!(format!("{}", z1), "1 + 2i");

        let z2 = Complex::new(1.0, -2.0);
        assert_eq!(format!("{}", z2), "1 - 2i");

        let z3 = Complex::new(0.0, 2.0);
        assert_eq!(format!("{}", z3), "0 + 2i");
    }

    mod complex_arithmetic_tests {
        use super::*;

        #[test]
        fn test_add() {
            let a = Complex::new(1.0, 2.0);
            let b = Complex::new(3.0, 4.0);
            let c = a + b;
            assert_eq!(c, Complex::new(4.0, 6.0));
        }

        #[test]
        fn test_sub() {
            let a = Complex::new(3.0, 4.0);
            let b = Complex::new(1.0, 2.0);
            let c = a - b;
            assert_eq!(c, Complex::new(2.0, 2.0));
        }

        #[test]
        fn test_mul() {
            let a = Complex::new(1.0, 2.0);
            let b = Complex::new(3.0, 4.0);
            let c = a * b;
            assert_eq!(c, Complex::new(-5.0, 10.0));
        }

        #[test]
        fn test_div() {
            let a = Complex::new(1.0, 2.0);
            let b = Complex::new(3.0, 4.0);
            let c = a / b;
            assert_eq!(c, Complex::new(11.0 / 25.0, 2.0 / 25.0));
        }

        #[test]
        fn test_neg() {
            let a = Complex::new(1.0, 2.0);
            let b = -a;
            assert_eq!(b, Complex::new(-1.0, -2.0));
        }
    }

    mod vector_complex_tests {
        use super::*;

        #[test]
        fn test_vector_add() {
            let mut v1 = Vector::from([Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
            let v2 = Vector::from([Complex::new(3.0, 3.0), Complex::new(4.0, 4.0)]);
            v1.add(&v2);
            assert_eq!(
                v1.data,
                vec![Complex::new(4.0, 4.0), Complex::new(6.0, 6.0)]
            );
        }

        #[test]
        fn test_vector_dot() {
            let v1 = Vector::from([Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);
            let v2 = Vector::from([Complex::new(3.0, 3.0), Complex::new(4.0, 4.0)]);
            let dot = v1.dot(&v2);
            assert_eq!(dot, Complex::new(0.0, 22.0));
        }

        #[test]
        fn test_vector_norm() {
            let v = Vector::from([Complex::new(3.0, 4.0)]);
            assert_eq!(v.norm(), 5.0);
        }
    }

    mod matrix_complex_tests {
        use super::*;

        #[test]
        fn test_matrix_mul() {
            let m1 = Matrix::from([
                [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
                [Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)],
            ]);
            let m2 = Matrix::from([
                [Complex::new(0.0, 1.0), Complex::new(0.0, 2.0)],
                [Complex::new(0.0, 3.0), Complex::new(0.0, 4.0)],
            ]);
            let result = m1.mul_mat(&m2);
            assert_eq!(result.data[0][0], Complex::new(0.0, 7.0));
            assert_eq!(result.data[0][1], Complex::new(0.0, 10.0));
            assert_eq!(result.data[1][0], Complex::new(0.0, 15.0));
            assert_eq!(result.data[1][1], Complex::new(0.0, 22.0));
        }

        #[test]
        fn test_matrix_determinant() {
            let m = Matrix::from([
                [Complex::new(1.0, 1.0), Complex::new(0.0, 1.0)],
                [Complex::new(1.0, 0.0), Complex::new(1.0, 1.0)],
            ]);
            let det = m.determinant();
            assert_eq!(det, Complex::new(0.0, 1.0));
        }
    }

    mod complex_operations_tests {
        use super::*;

        #[test]
        fn test_linear_combination() {
            let v1 = Vector::from([Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)]);
            let v2 = Vector::from([Complex::new(0.0, 1.0), Complex::new(1.0, 0.0)]);
            let vectors = [v1, v2];
            let coefs = [Complex::new(1.0, 1.0), Complex::new(2.0, 0.0)];
            let result = linear_combination(&vectors, &coefs);
            assert_eq!(result.data[0], Complex::new(1.0, 3.0));
            assert_eq!(result.data[1], Complex::new(1.0, 1.0));
        }

        #[test]
        fn test_cross_product() {
            let v1 = Vector::from([
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ]);
            let v2 = Vector::from([
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ]);
            let result = cross_product(&v1, &v2);
            assert_eq!(result.data[2], Complex::new(1.0, 0.0));
        }

        #[test]
        fn test_angle_cos() {
            let v1 = Vector::from([Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
            let v2 = Vector::from([Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)]);
            let cos = angle_cos(&v1, &v2);
            assert!(cos == 1.0);
        }
    }
}
