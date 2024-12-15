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
}
