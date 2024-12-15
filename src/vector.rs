//! Vector type implementation for linear algebra operations.
//!
//! Provides a generic Vector type that works with any Scalar type.
//! Includes basic vector operations and utilities.

use crate::Scalar;
use std::fmt::{Display, Formatter};

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
pub struct Vector<K: Scalar> {
    data: Vec<K>,
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
}

// Implementation for initializing a Vector with an array
impl<K: Scalar, const SIZE: usize> From<[K; SIZE]> for Vector<K> {
    fn from(data: [K; SIZE]) -> Self {
        Self {
            data: data.to_vec(),
        }
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
        fn test_vector_zero_scale() {
            let mut v = create_test_vector();
            v.scl(0.0);
            assert_eq!(v.data, vec![0.0, 0.0, 0.0]);
        }
    }
}
