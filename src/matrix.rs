//! Matrix type implementation for linear algebra operations.
//!
//! Provides a generic Matrix type that works with any Scalar type.
//! Includes basic matrix operations and utilities.

use crate::Scalar;
use std::fmt::{Display, Formatter};

/// A matrix of scalar values.
///
/// Represents a mathematical matrix as a two-dimensional array of scalar values.
/// The matrix stores its elements in a vector of vectors.
///
/// # Example
/// ```
/// use matrix::Matrix;
///
/// // Create a 2x2 matrix
/// let m = Matrix::from([
///     [1.0, 2.0],
///     [3.0, 4.0]
/// ]);
/// ```
pub struct Matrix<K: Scalar> {
    data: Vec<Vec<K>>,
}

impl<K: Scalar> Matrix<K> {
    /// Creates a `Matrix<K>` from a vector of vectors of Scalar `K` values.
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let mat = Matrix::new(vec![
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0]
    /// ]);
    /// ```
    pub fn new(data: Vec<Vec<K>>) -> Self {
        Self { data }
    }

    /// Returns the shape of the matrix as a tuple (rows, columns).
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// let (rows, cols) = m.shape(); // Returns (2, 2)
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Returns the number of rows in the matrix.
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// let rows = m.rows(); // Returns 2
    /// ```
    pub fn rows(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of columns in the matrix.
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// let cols = m.cols(); // Returns 2
    /// ```
    pub fn cols(&self) -> usize {
        if self.rows() > 0 {
            self.data[0].len()
        } else {
            0
        }
    }

    /// Prints the contents of the `Matrix` to standard output.
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// m.print();
    ///
    /// // Outputs:
    /// // [1.0, 2.0]
    /// // [3.0, 4.0]
    /// ```
    pub fn print(&self) {
        println!("{}", self);
    }

    /// Returns true if the matrix is square (same number of rows and columns).
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let square = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// let rect = Matrix::from([
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ]);
    ///
    /// square.is_square();    // true
    /// rect.is_square();      // false
    /// ```
    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }
}

// Implementation for initializing a Matrix with arrays
impl<K: Scalar, const COLS: usize, const ROWS: usize> From<[[K; COLS]; ROWS]> for Matrix<K> {
    fn from(data: [[K; COLS]; ROWS]) -> Self {
        Self {
            data: data.iter().map(|row| row.to_vec()).collect(),
        }
    }
}

// For displaying matrices
impl<K: Scalar> Display for Matrix<K> {
    /// Formats the `Matrix<K>` for display.
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// println!("{}", m);
    ///
    /// // Outputs:
    /// // [1.0, 2.0]
    /// // [3.0, 4.0]
    /// ```
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        // Helper closure to print the start of a row
        let print_row_start = |f: &mut Formatter, row: &Vec<K>| -> std::fmt::Result {
            write!(f, "[")?;
            let mut first = true;
            for val in row {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{}", val)?;
                first = false;
            }
            Ok(())
        };

        for row in &self.data[..self.rows() - 1] {
            print_row_start(f, row)?;
            writeln!(f, "]")?;
        }
        print_row_start(f, &self.data[self.rows() - 1])?;
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper functions to create test matrices
    fn create_test_matrix() -> Matrix<f32> {
        Matrix::from([[1.0, 2.0], [3.0, 4.0]])
    }
    fn create_test_rect_matrix() -> Matrix<f32> {
        Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    }

    mod construction_tests {
        use super::*;

        #[test]
        fn test_new() {
            let m = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
            assert_eq!(m.shape(), (2, 2));
        }

        #[test]
        fn test_from() {
            let data = [[1.0, 2.0], [3.0, 4.0]];
            let m = Matrix::from(data);
            assert_eq!(m.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        }
    }

    mod dimension_tests {
        use super::*;

        #[test]
        fn test_shape() {
            let m = create_test_rect_matrix();
            assert_eq!(m.shape(), (2, 3));
        }

        #[test]
        fn test_rows() {
            let m = create_test_rect_matrix();
            assert_eq!(m.rows(), 2);
        }

        #[test]
        fn test_cols() {
            let m = create_test_rect_matrix();
            assert_eq!(m.cols(), 3);
        }

        #[test]
        fn test_empty_matrix() {
            let m: Matrix<f32> = Matrix::new(vec![]);
            assert_eq!(m.rows(), 0);
            assert_eq!(m.cols(), 0);
        }

        #[test]
        fn test_is_square() {
            let square = create_test_matrix();
            let rect = create_test_rect_matrix();

            assert!(square.is_square());
            assert!(!rect.is_square());
        }
    }

    mod display_tests {
        use super::*;

        #[test]
        fn test_display() {
            let m = create_test_matrix();
            let display = format!("{}", m);
            assert!(display.contains("[1, 2]"));
            assert!(display.contains("[3, 4]"));
        }
    }
}
