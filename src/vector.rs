//! Vector type implementation for linear algebra operations.
//!
//! Provides a generic Vector type that works with any Scalar type.
//! Includes basic vector operations and utilities.

use crate::Scalar;
use std::fmt::{Display, Formatter};
use std::ops::{AddAssign, MulAssign, SubAssign};

/// A vector of scalar values.
///
/// Represents a mathematical vector as a sequence of scalar values.
/// The vector stores its elements in a contiguous array.
///
/// # Example
/// ```
/// use matrix::Vector;
///
/// // Create a vector from a Vec
/// let v = Vector::from([1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<K: Scalar> {
    /// The data stored in the Vector as a `Vec` of Scalar values.
    pub data: Vec<K>,
}

impl<K: Scalar> Vector<K> {
    /// Creates a `Vector<K>` from a vector of Scalar `K` values.
    ///
    /// # Example
    /// ```rust
    /// use matrix::Vector;
    ///
    /// let vec = Vector::new(vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn new(data: Vec<K>) -> Self {
        Self { data }
    }

    /// Returns the number of elements in the vector.
    ///
    /// # Example
    /// ```
    /// use matrix::Vector;
    ///
    /// let v = Vector::from([1.0, 2.0, 3.0]);
    /// let size = v.size(); // Returns 3
    /// ```
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Prints the contents of the `Vector` to standard output.
    ///
    /// # Example
    /// ```rust
    /// use matrix::Vector;
    ///
    /// let vec = Vector::from([1.0, 2.0, 3.0]);
    /// vec.print();
    ///
    /// // Outputs:
    /// // [1.0]
    /// // [2.0]
    /// // [3.0]
    /// ```
    pub fn print(&self) {
        println!("{}", self);
    }

    /// Adds another vector to this one in place.
    /// Both vectors must have the same size.
    ///
    /// # Complexity
    /// - Time: O(n) where n is the vector length - single pass over elements
    /// - Space: O(1) - no additional space allocated
    ///
    /// # Arguments
    /// * `other` - The vector to add to this one.
    ///
    /// # Panics
    /// Panics if the vectors have different sizes.
    ///
    /// # Example
    /// ```
    /// use matrix::Vector;
    ///
    /// let mut v1 = Vector::from([1.0, 2.0]);
    /// let v2 = Vector::from([3.0, 4.0]);
    /// v1.add(&v2);
    /// // v1 is now [4.0, 6.0]
    /// ```
    pub fn add(&mut self, other: &Vector<K>) {
        if self.size() != other.size() {
            panic!("Addition requires vectors of the same size.");
        }

        for (i, val) in other.data.iter().enumerate() {
            self.data[i] += *val;
        }
    }

    /// Subtracts another vector from this one in place.
    /// Both vectors must have the same size.
    ///
    /// # Complexity
    /// - Time: O(n) where n is the vector length - single pass over elements
    /// - Space: O(1) - no additional space allocated
    ///
    /// # Arguments
    /// * `other` - The vector to subtract from this one.
    ///
    /// # Panics
    /// Panics if the vectors have different sizes.
    ///
    /// # Example
    /// ```
    /// use matrix::Vector;
    ///
    /// let mut v1 = Vector::from([3.0, 4.0]);
    /// let v2 = Vector::from([1.0, 2.0]);
    /// v1.sub(&v2);
    /// // v1 is now [2.0, 2.0]
    /// ```
    pub fn sub(&mut self, other: &Vector<K>) {
        if self.size() != other.size() {
            panic!("Subtraction requires vectors of the same size.");
        }

        for (i, val) in other.data.iter().enumerate() {
            self.data[i] -= *val;
        }
    }

    /// Scales this vector by a scalar value in place.
    ///
    /// # Complexity
    /// - Time: O(n) where n is the vector length - single pass over elements
    /// - Space: O(1) - no additional space allocated
    ///
    /// # Arguments
    /// * `a` - The scalar value to scale the vector by.
    ///
    /// # Example
    /// ```
    /// use matrix::Vector;
    ///
    /// let mut v = Vector::from([1.0, 2.0]);
    /// v.scl(2.0);
    /// // v is now [2.0, 4.0]
    /// ```
    pub fn scl(&mut self, a: K) {
        for val in &mut self.data {
            *val *= a;
        }
    }

    /// Computes the dot product (inner product) with another vector.
    ///
    /// The dot product between vectors u and v is defined as:
    /// `sum(u[i] * v[i]) for i = 0 to n-1`
    ///
    /// # Complexity
    /// - Time: O(n) where n is the vector length - single pass over elements
    /// - Space: O(1) - only stores the accumulator
    ///
    /// # Performance Note
    /// Uses FMA (Fused Multiply-Add) operations for better numerical precision.
    ///
    /// # Arguments
    /// * `other` - The vector to compute dot product with
    ///
    /// # Panics
    /// Panics if vectors have different sizes
    ///
    /// # Example
    /// ```
    /// use matrix::Vector;
    ///
    /// let v1 = Vector::from([1.0, 2.0]);
    /// let v2 = Vector::from([3.0, 4.0]);
    /// let dot = v1.dot(&v2); // 1.0*3.0 + 2.0*4.0 = 11.0
    /// ```
    pub fn dot(&self, other: &Vector<K>) -> K {
        if self.size() != other.size() {
            panic!("Dot product requires vectors of the same size");
        }

        let mut result = K::zero();
        for (x, y) in self.data.iter().zip(other.data.iter()) {
            // Use FMA: result = x * y + result
            result = K::fma(*x, *y, result);
        }
        result
    }

    /// Computes the Manhattan norm (1-norm) of this vector.
    ///
    /// The Manhattan norm is the sum of the absolute values of vector components:
    /// ∥v∥₁ = Σ|vᵢ|
    ///
    /// # Complexity
    /// - Time: O(n) where n is vector length - single pass over elements
    /// - Space: O(1) - only accumulator storage needed
    ///
    /// # Example
    /// ```
    /// use matrix::Vector;
    ///
    /// let v = Vector::from([3.0, -4.0]);
    /// let norm = v.norm_1(); // Returns 7.0 (|3| + |-4|)
    /// ```
    pub fn norm_1(&self) -> f32 {
        let mut sum: f32 = 0.0;
        for val in &self.data {
            sum += (*val).to_f32().abs();
        }
        sum
    }

    /// Computes the Euclidean norm (2-norm) of this vector.
    ///
    /// The Euclidean norm is the square root of the sum of squared components:
    /// ∥v∥₂ = √(Σvᵢ²)
    ///
    /// Uses FMA for better numerical precision when computing sum of squares.
    ///
    /// # Complexity
    /// - Time: O(n) where n is vector length - single pass over elements
    /// - Space: O(1) - only accumulator storage needed
    ///
    /// # Example
    /// ```
    /// use matrix::Vector;
    ///
    /// let v = Vector::from([3.0, 4.0]);
    /// let norm = v.norm(); // Returns 5.0 (√(3² + 4²))
    /// ```
    pub fn norm(&self) -> f32 {
        let mut sum_sq: f32 = 0.0;
        for val in &self.data {
            let val_f32: f32 = (*val).to_f32();
            // Use FMA for sum of squares: val² + previous_sum
            sum_sq = f32::fma(val_f32, val_f32, sum_sq);
        }
        sum_sq.sqrt()
    }

    /// Computes the supremum norm (infinity norm) of this vector.
    ///
    /// The supremum norm is the maximum absolute value of vector components:
    /// ∥v∥∞ = max(|vᵢ|)
    ///
    /// # Complexity
    /// - Time: O(n) where n is vector length - single pass over elements
    /// - Space: O(1) - only stores current maximum
    ///
    /// # Example
    /// ```
    /// use matrix::Vector;
    ///
    /// let v = Vector::from([1.0, -5.0, 3.0]);
    /// let norm = v.norm_inf(); // Returns 5.0 (max(|1|, |-5|, |3|))
    /// ```
    pub fn norm_inf(&self) -> f32 {
        self.data
            .iter()
            .map(|val| (*val).to_f32().abs())
            .fold(0.0, f32::max)
    }
}

/// Computes a linear combination of vectors.
///
/// # Complexity
/// - Time: O(n) where n is the vector length - single pass over elements
/// - Space: O(n) - allocates a new result vector
///
/// # Arguments
/// * `u` - Array slice of vectors
/// * `coefs` - Array slice of coefficients corresponding to each vector
///
/// # Panics
/// - If the number of vectors and coefficients don't match
/// - If vectors have different sizes
///
/// # Example
/// ```
/// use matrix::{Vector, linear_combination};
///
/// let v1 = Vector::from([1.0, 2.0]);
/// let v2 = Vector::from([3.0, 4.0]);
/// let vectors = [v1, v2];
/// let coefs = [2.0, -1.0];
///
/// let result = linear_combination(&vectors, &coefs);
/// // result is [-1.0, 0.0] = 2.0 * [1.0, 2.0] + (-1.0) * [3.0, 4.0]
/// ```
pub fn linear_combination<K: Scalar>(u: &[Vector<K>], coefs: &[K]) -> Vector<K> {
    // Verify inputs
    if u.is_empty() || coefs.is_empty() || u.len() != coefs.len() {
        panic!("Linear combination requires equal non-zero number of vectors and coefficients");
    }

    let size = u[0].size();
    if !u.iter().all(|v| v.size() == size) {
        panic!("All vectors must have the same size");
    }

    // Initialize result vector with zeros
    let mut result = vec![K::zero(); size];

    // Use FMA for more accurate accumulation if available
    for (vector, &coef) in u.iter().zip(coefs.iter()) {
        for (r, &el) in result.iter_mut().zip(vector.data.iter()) {
            *r = K::fma(coef, el, *r); // coef * el + previous_result
        }
    }

    Vector::new(result)
}

/// Computes the cosine of the angle between two vectors.
///
/// The cosine is calculated using the formula:
/// cos(θ) = (u·v) / (∥u∥∥v∥)
/// where:
/// - u·v is the dot product
/// - ∥u∥ is the Euclidean norm (2-norm)
///
/// # Complexity
/// - Time: O(n) where n is the vector length - one pass for dot product and two for norms
/// - Space: O(1) - only stores temporary calculations
///
/// # Arguments
/// * `u` - First vector
/// * `v` - Second vector
///
/// # Panics
/// - If vectors have different sizes
/// - If either vector is zero (undefined angle)
///
/// # Example
/// ```
/// use matrix::{Vector, angle_cos};
///
/// let v1 = Vector::from([1.0, 0.0]);
/// let v2 = Vector::from([0.0, 1.0]);
/// let cos = angle_cos(&v1, &v2); // Returns 0.0 (90 degrees)
/// ```
pub fn angle_cos<K: Scalar>(u: &Vector<K>, v: &Vector<K>) -> f32 {
    if u.size() != v.size() {
        panic!("Vectors must have same size");
    }

    let dot = u.dot(v);
    let norm_u = u.norm();
    let norm_v = v.norm();

    if norm_u == f32::zero() || norm_v == f32::zero() {
        panic!("Zero vectors have undefined angle");
    }

    dot.to_f32() / (norm_u * norm_v)
}

/// Computes the cross product of two 3D vectors.
///
/// The cross product u × v is defined as:
/// ```txt
/// [u₂v₃ - u₃v₂, u₃v₁ - u₁v₃, u₁v₂ - u₂v₁]
/// ```
/// where u = [u₁, u₂, u₃] and v = [v₁, v₂, v₃]
///
/// The resulting vector is perpendicular to both input vectors
/// with magnitude equal to the area of the parallelogram they span.
///
/// # Arguments
/// * `u` - First 3D vector
/// * `v` - Second 3D vector
///
/// # Panics
/// * If either vector is not 3D
///
/// # Example
/// ```
/// use matrix::{Vector, cross_product};
///
/// let u = Vector::from([1.0, 0.0, 0.0]);  // i unit vector
/// let v = Vector::from([0.0, 1.0, 0.0]);  // j unit vector
/// let w = cross_product(&u, &v);          // k unit vector [0.0, 0.0, 1.0]
/// ```
pub fn cross_product<K: Scalar>(u: &Vector<K>, v: &Vector<K>) -> Vector<K> {
    if u.size() != 3 || v.size() != 3 {
        panic!("Cross product requires 3D vectors");
    }

    // Get components using array indexing
    let (u1, u2, u3) = (u.data[0], u.data[1], u.data[2]);
    let (v1, v2, v3) = (v.data[0], v.data[1], v.data[2]);

    // Calculate components using FMA where possible
    let x = K::fma(u2, v3, -(u3 * v2));
    let y = K::fma(u3, v1, -(u1 * v3));
    let z = K::fma(u1, v2, -(u2 * v1));

    Vector::from([x, y, z])
}

// Implementation for initializing a Vector with an array
impl<K: Scalar, const SIZE: usize> From<[K; SIZE]> for Vector<K> {
    fn from(data: [K; SIZE]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
}

// Implementations for vector assign multiplication, needed for lerp
impl<K: Scalar> MulAssign<K> for Vector<K> {
    fn mul_assign(&mut self, rhs: K) {
        self.scl(rhs);
    }
}

// Implementations for vector assign addition, needed for lerp
impl<K: Scalar> AddAssign for Vector<K> {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs);
    }
}

// Implementations for vector assign subtraction, needed for lerp
impl<K: Scalar> SubAssign for Vector<K> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(&rhs);
    }
}

// Display implementation for Vector
impl<K: Scalar> Display for Vector<K> {
    /// Formats the `Vector<K>` for display.
    ///
    /// # Example
    /// ```rust
    /// use matrix::Vector;
    ///
    /// let v = Vector::from([1., 2.]);
    /// println!("{}", v);
    ///
    /// // Outputs:
    /// // [1]
    /// // [2]
    /// ```
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        for val in &self.data[..self.size() - 1] {
            writeln!(f, "[{}]", val)?;
        }
        write!(f, "[{}]", self.data[self.size() - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create test vectors
    fn create_test_vector() -> Vector<f32> {
        Vector::from([1.0, 2.0, 3.0])
    }

    mod construction_tests {
        use super::*;

        #[test]
        fn test_new() {
            let v: Vector<f32> = Vector::new(vec![1.0, 2.0, 3.0]);
            assert_eq!(v.size(), 3);
        }

        #[test]
        fn test_from() {
            let data = [1.0, 2.0, 3.0];
            let v = Vector::from(data);
            assert_eq!(v.data, data);
        }
    }

    mod access_tests {
        use super::*;

        #[test]
        fn test_size() {
            let v = create_test_vector();
            assert_eq!(v.size(), 3);
        }
    }

    mod display_tests {
        use super::*;

        #[test]
        fn test_display() {
            let v = create_test_vector();
            let display = format!("{}", v);
            assert!(display.contains("[1]"));
            assert!(display.contains("[2]"));
            assert!(display.contains("[3]"));
        }
    }

    mod operation_tests {
        use super::*;

        #[test]
        fn test_vector_add() {
            let mut v1 = create_test_vector();
            let v2 = Vector::from([4.0, 5.0, 6.0]);
            v1.add(&v2);
            assert_eq!(v1.data, vec![5.0, 7.0, 9.0]);
        }

        #[test]
        fn test_vector_add_assign() {
            let mut v1 = create_test_vector();
            let v2 = Vector::from([4.0, 5.0, 6.0]);
            v1 += v2;
            assert_eq!(v1.data, vec![5.0, 7.0, 9.0]);
        }

        #[test]
        #[should_panic(expected = "Addition requires vectors of the same size.")]
        fn test_vector_add_panic() {
            let mut v1 = create_test_vector();
            let v2 = Vector::from([4.0, 5.0]);
            v1.add(&v2);
        }

        mod evaluation_add_tests {
            use super::*;

            #[test]
            fn test_vector_add_zero() {
                let mut v1 = Vector::from([0, 0]);
                let v2 = Vector::from([0, 0]);
                v1.add(&v2);
                assert_eq!(v1.data, vec![0, 0]);
            }

            #[test]
            fn test_vector_add_unit() {
                let mut v1 = Vector::from([1, 0]);
                let v2 = Vector::from([0, 1]);
                v1.add(&v2);
                assert_eq!(v1.data, vec![1, 1]);
            }

            #[test]
            fn test_vector_add_same() {
                let mut v1 = Vector::from([1, 1]);
                let v2 = Vector::from([1, 1]);
                v1.add(&v2);
                assert_eq!(v1.data, vec![2, 2]);
            }

            #[test]
            fn test_vector_add_42() {
                let mut v1 = Vector::from([21, 21]);
                let v2 = Vector::from([21, 21]);
                v1.add(&v2);
                assert_eq!(v1.data, vec![42, 42]);
            }

            #[test]
            fn test_vector_add_opposite() {
                let mut v1 = Vector::from([-21, 21]);
                let v2 = Vector::from([21, -21]);
                v1.add(&v2);
                assert_eq!(v1.data, vec![0, 0]);
            }

            #[test]
            fn test_vector_add_long() {
                let mut v1 = Vector::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
                let v2 = Vector::from([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
                v1.add(&v2);
                assert_eq!(v1.data, vec![9; 10]);
            }
        }

        #[test]
        fn test_vector_sub() {
            let mut v1 = Vector::from([4.0, 5.0, 6.0]);
            let v2 = create_test_vector();
            v1.sub(&v2);
            assert_eq!(v1.data, vec![3.0, 3.0, 3.0]);
        }

        #[test]
        fn test_vector_sub_assign() {
            let mut v1 = Vector::from([4.0, 5.0, 6.0]);
            let v2 = create_test_vector();
            v1 -= v2;
            assert_eq!(v1.data, vec![3.0, 3.0, 3.0]);
        }

        #[test]
        #[should_panic(expected = "Subtraction requires vectors of the same size.")]
        fn test_vector_sub_panic() {
            let mut v1 = create_test_vector();
            let v2 = Vector::from([4.0, 5.0]);
            v1.sub(&v2);
        }

        mod evaluation_sub_tests {
            use super::*;

            #[test]
            fn test_vector_sub_zero() {
                let mut v1 = Vector::from([0, 0]);
                let v2 = Vector::from([0, 0]);
                v1.sub(&v2);
                assert_eq!(v1.data, vec![0, 0]);
            }

            #[test]
            fn test_vector_sub_unit() {
                let mut v1 = Vector::from([1, 0]);
                let v2 = Vector::from([0, 1]);
                v1.sub(&v2);
                assert_eq!(v1.data, vec![1, -1]);
            }

            #[test]
            fn test_vector_sub_same() {
                let mut v1 = Vector::from([1, 1]);
                let v2 = Vector::from([1, 1]);
                v1.sub(&v2);
                assert_eq!(v1.data, vec![0, 0]);
            }

            #[test]
            fn test_vector_sub_same_21() {
                let mut v1 = Vector::from([21, 21]);
                let v2 = Vector::from([21, 21]);
                v1.sub(&v2);
                assert_eq!(v1.data, vec![0, 0]);
            }

            #[test]
            fn test_vector_sub_42() {
                let mut v1 = Vector::from([-21, 21]);
                let v2 = Vector::from([21, -21]);
                v1.sub(&v2);
                assert_eq!(v1.data, vec![-42, 42]);
            }

            #[test]
            fn test_vector_sub_long() {
                let mut v1 = Vector::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
                let v2 = Vector::from([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
                v1.sub(&v2);
                assert_eq!(v1.data, vec![-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]);
            }
        }

        #[test]
        fn test_vector_scl() {
            let mut v = create_test_vector();
            v.scl(2.0);
            assert_eq!(v.data, vec![2.0, 4.0, 6.0]);
        }

        #[test]
        fn test_vector_mul_assign() {
            let mut v = create_test_vector();
            v *= 2.0;
            assert_eq!(v.data, vec![2.0, 4.0, 6.0]);
        }

        #[test]
        fn test_vector_zero_scale() {
            let mut v = create_test_vector();
            v.scl(0.0);
            assert_eq!(v.data, vec![0.0, 0.0, 0.0]);
        }

        mod evaluation_scl_tests {
            use super::*;

            #[test]
            fn test_vector_scl_zero() {
                let mut v = Vector::from([0, 0]);
                v.scl(1);
                assert_eq!(v.data, vec![0, 0]);
            }

            #[test]
            fn test_vector_scl_unit() {
                let mut v = Vector::from([1, 0]);
                v.scl(1);
                assert_eq!(v.data, vec![1, 0]);
            }

            #[test]
            fn test_vector_scl_same() {
                let mut v = Vector::from([1, 1]);
                v.scl(2);
                assert_eq!(v.data, vec![2, 2]);
            }

            #[test]
            fn test_vector_scl_same_21() {
                let mut v = Vector::from([21, 21]);
                v.scl(2);
                assert_eq!(v.data, vec![42, 42]);
            }

            #[test]
            fn test_vector_scl_42() {
                let mut v = Vector::from([42.0, 42.0]);
                v.scl(0.5);
                assert_eq!(v.data, vec![21.0, 21.0]);
            }
        }
    }

    mod linear_combination_tests {
        use super::*;

        #[test]
        fn test_linear_combination() {
            let v1 = Vector::from([1.0, 2.0]);
            let v2 = Vector::from([3.0, 4.0]);
            let vectors = [v1, v2];
            let coefs = [2.0, -1.0];

            let result = linear_combination(&vectors, &coefs);
            assert_eq!(result.data, vec![-1.0, 0.0]);
        }

        #[test]
        #[should_panic(
            expected = "Linear combination requires equal non-zero number of vectors and coefficients"
        )]
        fn test_linear_combination_empty() {
            let vectors: [Vector<f32>; 0] = [];
            let coefs: [f32; 0] = [];
            linear_combination(&vectors, &coefs);
        }

        #[test]
        #[should_panic(
            expected = "Linear combination requires equal non-zero number of vectors and coefficients"
        )]
        fn test_linear_combination_mismatched_lengths() {
            let v = Vector::from([1.0, 2.0]);
            let vectors = [v];
            let coefs = [1.0, 2.0];
            linear_combination(&vectors, &coefs);
        }

        #[test]
        #[should_panic(expected = "All vectors must have the same size")]
        fn test_linear_combination_different_sizes() {
            let v1 = Vector::from([1.0, 2.0]);
            let v2 = Vector::from([3.0]);
            let vectors = [v1, v2];
            let coefs = [1.0, 1.0];
            linear_combination(&vectors, &coefs);
        }

        mod evaluation_linear_combination_tests {
            use super::*;

            #[test]
            fn test_linear_combination_1x_2d_vector() {
                let v = Vector::from([-42., 42.]);
                let vectors = [v];
                let coefs = [-1.];
                let result = linear_combination(&vectors, &coefs);
                assert_eq!(result.data, vec![42., -42.]);
            }

            #[test]
            fn test_linear_combination_3x_1d_vector() {
                let v = Vector::from([-42.]);
                let vectors = [v.clone(), v.clone(), v.clone()];
                let coefs = [-1., 1., 0.];
                let result = linear_combination(&vectors, &coefs);
                assert_eq!(result.data, vec![0.]);
            }

            #[test]
            fn test_linear_combination_3x_2d_vector() {
                let v1 = Vector::from([-42., 42.]);
                let v2 = Vector::from([1., 3.]);
                let v3 = Vector::from([10., 20.]);
                let vectors = [v1, v2, v3];
                let coefs = [1., -10., -1.];
                let result = linear_combination(&vectors, &coefs);
                assert_eq!(result.data, vec![-62., -8.]);
            }

            #[test]
            fn test_linear_combination_2x_3d_vector() {
                let v1 = Vector::from([-42., 100., -69.5]);
                let v2 = Vector::from([1., 3., 5.]);
                let vectors = [v1, v2];
                let coefs = [1., -10.];
                let result = linear_combination(&vectors, &coefs);
                assert_eq!(result.data, vec![-52., 70., -119.5]);
            }
        }
    }

    mod dot_product_tests {
        use super::*;

        #[test]
        fn test_dot_product() {
            let v1 = Vector::from([1.0, 2.0, 3.0]);
            let v2 = Vector::from([4.0, 5.0, 6.0]);
            assert_eq!(v1.dot(&v2), 32.0); // 1*4 + 2*5 + 3*6 = 32
        }

        #[test]
        fn test_dot_product_orthogonal() {
            let v1 = Vector::from([1.0, 0.0]);
            let v2 = Vector::from([0.0, 1.0]);
            assert_eq!(v1.dot(&v2), 0.0); // Orthogonal vectors have dot product 0
        }

        #[test]
        fn test_dot_product_parallel() {
            let v1 = Vector::from([2.0, 0.0]);
            let v2 = Vector::from([3.0, 0.0]);
            assert_eq!(v1.dot(&v2), 6.0); // Parallel vectors scale dot product
        }

        #[test]
        #[should_panic(expected = "Dot product requires vectors of the same size")]
        fn test_dot_product_different_sizes() {
            let v1 = Vector::from([1.0, 2.0]);
            let v2 = Vector::from([1.0, 2.0, 3.0]);
            v1.dot(&v2);
        }

        mod evaluation_dot_product_tests {
            use super::*;

            #[test]
            fn test_dot_product_zero() {
                let v1 = Vector::from([0, 0]);
                let v2 = Vector::from([0, 0]);
                assert_eq!(v1.dot(&v2), 0);
            }

            #[test]
            fn test_dot_product_unit() {
                let v1 = Vector::from([1, 0]);
                let v2 = Vector::from([0, 0]);
                assert_eq!(v1.dot(&v2), 0);
            }

            #[test]
            fn test_dot_product_same() {
                let v1 = Vector::from([1, 0]);
                let v2 = Vector::from([1, 0]);
                assert_eq!(v1.dot(&v2), 1);
            }

            #[test]
            fn test_dot_product_orthogonal() {
                let v1 = Vector::from([1, 0]);
                let v2 = Vector::from([0, 1]);
                assert_eq!(v1.dot(&v2), 0);
            }

            #[test]
            fn test_dot_product_same_2() {
                let v1 = Vector::from([1, 1]);
                let v2 = Vector::from([1, 1]);
                assert_eq!(v1.dot(&v2), 2);
            }

            #[test]
            fn test_dot_product_long() {
                let v1 = Vector::from([4, 2]);
                let v2 = Vector::from([2, 1]);
                assert_eq!(v1.dot(&v2), 10);
            }
        }
    }

    #[cfg(test)]
    mod norms_tests {
        use super::*;

        #[test]
        fn test_norm_1() {
            let v = Vector::from([1.0, -2.0, 3.0]);
            assert_eq!(v.norm_1(), 6.0);

            let v = Vector::from([-1.0, -1.0]);
            assert_eq!(v.norm_1(), 2.0);

            let v = Vector::from([0.0, 0.0, 0.0]);
            assert_eq!(v.norm_1(), 0.0);
        }

        #[test]
        fn test_norm_2() {
            let v = Vector::from([3.0, 4.0]);
            assert_eq!(v.norm(), 5.0);

            let v = Vector::from([1.0, 1.0, 1.0]);
            assert!((v.norm() - 3_f32.sqrt()).abs() < f32::EPSILON);

            let v = Vector::from([0.0, 0.0]);
            assert_eq!(v.norm(), 0.0);
        }

        #[test]
        fn test_norm_inf() {
            let v = Vector::from([1.0, -5.0, 3.0]);
            assert_eq!(v.norm_inf(), 5.0);

            let v = Vector::from([-2.0, 2.0]);
            assert_eq!(v.norm_inf(), 2.0);

            let v = Vector::from([0.0, 0.0, 0.0]);
            assert_eq!(v.norm_inf(), 0.0);
        }

        #[test]
        fn test_relationship_between_norms() {
            // For any vector v in R^n: ∥v∥∞ ≤ ∥v∥₂ ≤ ∥v∥₁
            let v = Vector::from([1.0, 2.0, -3.0]);

            let norm_inf = v.norm_inf();
            let norm_2 = v.norm();
            let norm_1 = v.norm_1();

            assert!(norm_inf <= norm_2);
            assert!(norm_2 <= norm_1);
        }
    }

    mod cosine_tests {
        use super::*;

        #[test]
        fn test_cosine_parallel() {
            let v1 = Vector::from([2.0, 0.0]);
            let v2 = Vector::from([4.0, 0.0]);
            assert!((angle_cos(&v1, &v2) - 1.0).abs() < f32::EPSILON);
        }

        #[test]
        fn test_cosine_perpendicular() {
            let v1 = Vector::from([1.0, 0.0]);
            let v2 = Vector::from([0.0, 1.0]);
            assert!(angle_cos(&v1, &v2).abs() < f32::EPSILON);
        }

        #[test]
        fn test_cosine_opposite() {
            let v1 = Vector::from([1.0, 0.0]);
            let v2 = Vector::from([-1.0, 0.0]);
            assert!((angle_cos(&v1, &v2) - (-1.0)).abs() < f32::EPSILON);
        }

        #[test]
        fn test_cosine_45_degrees() {
            let v1 = Vector::from([1.0, 0.0]);
            let v2 = Vector::from([1.0, 1.0]);
            assert!((angle_cos(&v1, &v2) - 0.7071067812).abs() < 1e-7);
        }

        #[test]
        #[should_panic(expected = "Zero vectors have undefined angle")]
        fn test_cosine_zero_vector() {
            let v1 = Vector::from([0.0, 0.0]);
            let v2 = Vector::from([1.0, 0.0]);
            angle_cos(&v1, &v2);
        }

        #[test]
        #[should_panic(expected = "Vectors must have same size")]
        fn test_cosine_different_sizes() {
            let v1 = Vector::from([1.0, 0.0]);
            let v2 = Vector::from([1.0]);
            angle_cos(&v1, &v2);
        }
    }

    mod cross_product_tests {
        use super::*;

        #[test]
        fn test_cross_unit_vectors() {
            // i × j = k
            let i = Vector::from([1.0, 0.0, 0.0]);
            let j = Vector::from([0.0, 1.0, 0.0]);
            let k = Vector::from([0.0, 0.0, 1.0]);

            assert_eq!(cross_product(&i, &j).data, k.data);
            // j × k = i
            assert_eq!(cross_product(&j, &k).data, i.data);
            // k × i = j
            assert_eq!(cross_product(&k, &i).data, j.data);
        }

        #[test]
        fn test_cross_anticommutative() {
            // u × v = -(v × u)
            let u = Vector::from([2.0, 3.0, 4.0]);
            let v = Vector::from([5.0, 6.0, 7.0]);

            let uv = cross_product(&u, &v);
            let vu = cross_product(&v, &u);

            for (a, b) in uv.data.iter().zip(vu.data.iter()) {
                assert!((*a + *b).to_f32().abs() < f32::EPSILON);
            }
        }

        #[test]
        fn test_cross_orthogonal() {
            // (u × v) · u = 0 and (u × v) · v = 0
            let u = Vector::from([1.0, 2.0, 3.0]);
            let v = Vector::from([4.0, 5.0, 6.0]);
            let expected = Vector::from([-3.0, 6.0, -3.0]);
            let result = cross_product(&u, &v);
            assert_eq!(result.data, expected.data);
        }

        #[test]
        fn test_cross_example() {
            let u = Vector::from([4.0, 2.0, -3.0]);
            let v = Vector::from([-2.0, -5.0, 16.0]);
            let expected = Vector::from([17.0, -58.0, -16.0]);
            let result = cross_product(&u, &v);
            assert_eq!(result.data, expected.data);
        }

        #[test]
        #[should_panic(expected = "Cross product requires 3D vectors")]
        fn test_cross_wrong_dimension() {
            let u = Vector::from([1.0, 0.0]);
            let v = Vector::from([0.0, 1.0]);
            cross_product(&u, &v);
        }
    }
}
