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
#[derive(Debug, Clone)]
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
            sum += Into::<f32>::into(*val).abs();
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
            let val_f32: f32 = (*val).into();
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
            .map(|val| Into::<f32>::into(*val).abs())
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
    }

    mod dot_product_test {
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
}
