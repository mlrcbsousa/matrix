//! Matrix type implementation for linear algebra operations.
//!
//! Provides a generic Matrix type that works with any Scalar type.
//! Includes basic matrix operations and utilities.

use crate::{Scalar, Vector};
use std::fmt::{Display, Formatter};
use std::ops::{AddAssign, MulAssign, SubAssign};

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
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<K: Scalar> {
    /// The data of the matrix stored as a vector of vectors.
    pub data: Vec<Vec<K>>,
}

/// Error type for matrix inverse operations
#[derive(Debug, PartialEq)]
pub enum MatrixError {
    /// Matrix is singular (non-invertible)
    Singular,
    /// Matrix is not square
    NotSquare,
}

impl<K: Scalar> Matrix<K> {
    /// Numerical tolerance for considering values effectively zero
    /// Used across multiple matrix operations (RREF, inverse, etc.)
    ///
    /// We could use f32::EPSILON (≈1.19e-7) which represents the smallest positive
    /// number where 1.0 + x ≠ 1.0 in f32. However, for RREF we chose a smaller
    /// tolerance (1e-10) because:
    /// 1. In RREF, we want to be very certain a value is truly zero before treating
    ///    it as a pivot or declaring linear dependence
    /// 2. When solving systems of equations, false non-zeros can lead to incorrect
    ///    rank calculations and thus wrong solutions
    /// 3. The added precision helps distinguish nearly-dependent rows that are
    ///    actually independent
    ///
    /// Note: If numerical stability becomes an issue (e.g., with very large matrices
    /// or ill-conditioned systems), increasing this to f32::EPSILON might be appropriate
    const TOLERANCE: f32 = 1e-10;

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

    /// Returns true if the matrix is empty (zero rows or columns).
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let empty = Matrix::<f32>::new(vec![]);
    /// let non_empty = Matrix::from([[1.0]]);
    ///
    /// empty.is_empty();       // true
    /// non_empty.is_empty();   // false
    /// ```
    pub fn is_empty(&self) -> bool {
        self.rows() == 0 || self.cols() == 0
    }

    /// Adds another matrix to this one in place.
    /// Both matrices must have the same dimensions.
    ///
    /// # Complexity
    /// - Time: O(n) where n is the total number of elements (rows × columns) - single pass over elements
    /// - Space: O(1) - no additional space allocated
    ///
    /// # Arguments
    /// * `other` - The matrix to add to this one.
    ///
    /// # Panics
    /// Panics if the matrices have different dimensions.
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let mut m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
    /// m1.add(&m2);
    /// // m1 is now [[6.0, 8.0], [10.0, 12.0]]
    /// ```
    pub fn add(&mut self, other: &Matrix<K>) {
        if self.shape() != other.shape() {
            panic!("Addition requires matrices with the same shape.");
        }

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self.data[i][j] += other.data[i][j];
            }
        }
    }

    /// Subtracts another matrix from this one in place.
    /// Both matrices must have the same dimensions.
    ///
    /// # Complexity
    /// - Time: O(n) where n is the total number of elements (rows × columns) - single pass over elements
    /// - Space: O(1) - no additional space allocated
    ///
    /// # Arguments
    /// * `other` - The matrix to subtract from this one.
    ///
    /// # Panics
    /// Panics if the matrices have different dimensions.
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let mut m1 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
    /// let m2 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    /// m1.sub(&m2);
    /// // m1 is now [[4.0, 4.0], [4.0, 4.0]]
    /// ```
    pub fn sub(&mut self, other: &Matrix<K>) {
        if self.shape() != other.shape() {
            panic!("Subtraction requires matrices with the same shape.");
        }

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self.data[i][j] -= other.data[i][j];
            }
        }
    }

    /// Scales this matrix by a scalar value in place.
    ///
    /// # Complexity
    /// - Time: O(n) where n is the total number of elements (rows × columns) - single pass over elements
    /// - Space: O(1) - no additional space allocated
    ///
    /// # Arguments
    /// * `a` - The scalar value to scale the matrix by.
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let mut m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    /// m.scl(2.0);
    /// // m is now [[2.0, 4.0], [6.0, 8.0]]
    /// ```
    pub fn scl(&mut self, a: K) {
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                self.data[i][j] *= a;
            }
        }
    }

    /// Multiplies this matrix by a vector (matrix-vector multiplication).
    ///
    /// For a matrix A and vector x, computes Ax where:
    /// (Ax)ᵢ = Σⱼ(aᵢⱼxⱼ)
    ///
    /// # Complexity
    /// - Time: O(nm) where n = rows, m = cols
    /// - Space: O(nm) for storing result
    ///
    /// # Arguments
    /// * `vec` - Vector to multiply with (must have same size as matrix columns)
    ///
    /// # Panics
    /// * If matrix columns don't match vector size
    ///
    /// # Example
    /// ```
    /// use matrix::{Matrix, Vector};
    ///
    /// let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let v = Vector::from([1.0, 2.0]);
    /// let result = m.mul_vec(&v); // [5.0, 11.0]
    /// ```
    pub fn mul_vec(&self, vec: &Vector<K>) -> Vector<K> {
        if self.cols() != vec.size() {
            panic!("Matrix columns must match vector size");
        }

        let mut result = vec![K::zero(); self.rows()];

        // For each row of matrix
        for i in 0..self.rows() {
            // Compute product with vector
            for (j, v_j) in vec.data.iter().enumerate() {
                result[i] = K::fma(self.data[i][j], *v_j, result[i]);
            }
        }

        Vector::new(result)
    }

    /// Multiplies this matrix by another matrix (matrix-matrix multiplication).
    ///
    /// For matrices A and B, computes AB where:
    /// (AB)ᵢⱼ = Σₖ(aᵢₖbₖⱼ)
    ///
    /// # Complexity
    /// - Time: O(nmp) where:
    ///   - n = rows of first matrix
    ///   - m = cols of first matrix / rows of second matrix
    ///   - p = cols of second matrix
    /// - Space: O(nm + mp + np) for intermediate and result storage
    ///
    /// # Arguments
    /// * `other` - Matrix to multiply with (columns must match this matrix's rows)
    ///
    /// # Panics
    /// * If matrix dimensions don't match for multiplication
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
    /// let result = m1.mul_mat(&m2); // [[19.0, 22.0], [43.0, 50.0]]
    /// ```
    pub fn mul_mat(&self, other: &Matrix<K>) -> Matrix<K> {
        if self.cols() != other.rows() {
            panic!("First matrix columns must match second matrix rows");
        }

        let n = self.rows();
        let m = self.cols();
        let p = other.cols();

        let mut result = vec![vec![K::zero(); p]; n];

        // For each element of result matrix
        for i in 0..n {
            for j in 0..p {
                // Compute product of row i from first and col j from second
                for k in 0..m {
                    result[i][j] = K::fma(self.data[i][k], other.data[k][j], result[i][j]);
                }
            }
        }

        Matrix::new(result)
    }

    /// Computes the trace of this matrix.
    ///
    /// The trace is the sum of elements on the main diagonal:
    /// tr(A) = Σᵢaᵢᵢ
    ///
    /// # Complexity
    /// - Time: O(n) where n is the matrix dimension
    /// - Space: O(1) - only stores accumulator
    ///
    /// # Panics
    /// * If matrix is not square
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// let tr = m.trace(); // Returns 5.0 (1.0 + 4.0)
    /// ```
    pub fn trace(&self) -> K {
        if !self.is_square() {
            panic!("Trace only defined for square matrices");
        }

        let n = self.rows();
        let mut sum = K::zero();
        for i in 0..n {
            sum += self.data[i][i];
        }
        sum
    }

    /// Computes the transpose of this matrix.
    ///
    /// The transpose of a matrix A is formed by flipping matrix elements across its main diagonal:
    /// - Each element aᵢⱼ becomes aⱼᵢ in the transposed matrix
    /// - The shape changes from m×n to n×m
    ///
    /// # Complexity
    /// - Time: O(nm) where n = rows, m = cols - must visit each element once
    /// - Space: O(nm) - must allocate new matrix
    ///
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    ///
    /// // Square matrix
    /// let m = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// let m_t = m.transpose();
    /// // m_t is [[1.0, 3.0], [2.0, 4.0]]
    ///
    /// // Rectangular matrix
    /// let m = Matrix::from([
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0]
    /// ]);
    /// let m_t = m.transpose();
    /// // m_t is [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    /// ```
    pub fn transpose(&self) -> Matrix<K> {
        let rows = self.rows();
        let cols = self.cols();
        let mut result = vec![vec![K::zero(); rows]; cols];

        for i in 0..rows {
            for j in 0..cols {
                result[j][i] = self.data[i][j];
            }
        }

        Matrix::new(result)
    }

    /// Computes the Reduced Row Echelon Form (RREF) of this matrix using Gaussian elimination.
    ///
    /// Algorithm steps:
    /// 1. For each column (potential pivot):
    ///    a. Find row with largest absolute value in current column (partial pivoting)
    ///    b. If largest value < TOLERANCE, skip column (no pivot here)
    ///    c. Swap row with current pivot row
    ///    d. Scale pivot row to make pivot = 1
    ///    e. Eliminate entries in pivot column in all other rows
    /// 2. Clean up values smaller than TOLERANCE to exactly zero
    ///
    /// Uses partial pivoting and numerical tolerance for stability.
    ///
    /// # Complexity
    /// - Time: O(n³) where n is the largest dimension
    /// - Space: O(n²) for the result matrix
    ///
    /// # Examples
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [2.0, 4.0],
    ///     [1.0, 2.0]
    /// ]);
    /// let rref = m.row_echelon();
    /// // Result is:
    /// // [1.0, 2.0]
    /// // [0.0, 0.0]
    /// ```
    pub fn row_echelon(&self) -> Matrix<K> {
        let rows = self.rows();
        let cols = self.cols();

        if self.is_empty() {
            panic!("Cannot compute RREF of an empty matrix.");
        }

        for row in &self.data {
            if row.len() != cols {
                panic!("Matrix rows must have the same number of columns.");
            }
        }

        let mut result = self.clone();
        let mut pivot_row = 0;

        // Forward elimination to RREF
        for pivot_col in 0..cols {
            // Find row with largest absolute value in current column (partial pivoting)
            let mut max_row = pivot_row;
            let mut max_val = f32::zero();
            for r in pivot_row..rows {
                let val: f32 = result.data[r][pivot_col].to_f32();
                if val.abs() > max_val {
                    max_val = val.abs();
                    max_row = r;
                }
            }

            // Skip column if no valid pivot found
            if max_val < Self::TOLERANCE {
                continue;
            }

            // Swap max row to pivot position
            if max_row != pivot_row {
                result.data.swap(max_row, pivot_row);
            }

            // Scale pivot row to get leading 1
            let pivot = result.data[pivot_row][pivot_col];
            for c in pivot_col..cols {
                result.data[pivot_row][c] /= pivot;
            }

            // Eliminate non-zero entries in pivot column
            for r in 0..rows {
                if r != pivot_row {
                    let factor = result.data[r][pivot_col];
                    if factor.to_f32().abs() > Self::TOLERANCE {
                        for c in pivot_col..cols {
                            result.data[r][c] =
                                result.data[r][c] - factor * result.data[pivot_row][c];

                            // Clean up near-zero values
                            if result.data[r][c].to_f32().abs() < Self::TOLERANCE {
                                result.data[r][c] = K::zero();
                            }
                        }
                    }
                }
            }

            pivot_row += 1;
            if pivot_row == rows {
                break;
            }
        }

        result
    }

    /// Computes the determinant of the matrix.
    ///
    /// The determinant is a scalar value that provides information about:
    /// - Whether the matrix is invertible (det ≠ 0)
    /// - The scaling factor of linear transformation
    /// - The orientation/handedness change of transformation
    ///
    /// For matrices up to 4x4, uses dimension-specific optimized formulas:
    /// - 1x1: det = a
    /// - 2x2: det = ad - bc
    /// - 3x3: Uses cofactor expansion with 2x2 determinants
    /// - 4x4: Uses cofactor expansion with 3x3 determinants
    ///
    /// # Complexity
    /// - Time: O(n³) where n is matrix dimension
    /// - Space: O(n²) for recursive calculations
    ///
    /// # Panics
    /// - If matrix is not square
    /// - If matrix dimension > 4
    /// - If matrix has 0 rows (empty)
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    /// let m = Matrix::from([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    /// let det = m.determinant(); // Returns -2.0
    /// ```
    pub fn determinant(&self) -> K {
        if !self.is_square() {
            panic!("Determinant only defined for square matrices");
        }

        let n = self.rows();
        match n {
            0 => panic!("Empty matrix has no determinant"),
            1 => self.data[0][0],
            2 => self.det2x2(),
            3 => self.det3x3(),
            4 => self.det4x4(),
            _ => panic!("Determinant implementation limited to 4x4 matrices"),
        }
    }

    // Helper function for 2x2 determinant
    // Formula: |A| = ad - bc for matrix [[a b], [c d]]
    fn det2x2(&self) -> K {
        let a = self.data[0][0];
        let b = self.data[0][1];
        let c = self.data[1][0];
        let d = self.data[1][1];

        // Use FMA: ad - bc
        K::fma(a, d, -(b * c))
    }

    // Helper function for 3x3 determinant
    fn det3x3(&self) -> K {
        let mut det = K::zero();

        // Expand along first row using 2x2 determinants
        for j in 0..3 {
            // Get 2x2 minor by removing row 0 and column j
            let minor = self.get_minor(0, j);

            // Calculate sign: (-1)^(i+j) where i=0
            let sign = if j % 2 == 0 { K::one() } else { -K::one() };

            // Use det2x2() for minor and add to total
            det = K::fma(self.data[0][j] * sign, minor.det2x2(), det);
        }

        det
    }

    // Helper function for 4x4 determinant
    fn det4x4(&self) -> K {
        let mut det = K::zero();

        // Expand along first row using 3x3 determinants
        for j in 0..4 {
            // Get 3x3 minor by removing row 0 and column j
            let minor = self.get_minor(0, j);

            // Calculate sign: (-1)^(i+j) where i=0
            let sign = if j % 2 == 0 { K::one() } else { -K::one() };

            // Use det3x3() for minor and add to total
            det = K::fma(self.data[0][j] * sign, minor.det3x3(), det);
        }

        det
    }

    // Helper function to compute minor (determinant of submatrix)
    fn get_minor(&self, row: usize, col: usize) -> Matrix<K> {
        let n = self.rows();
        let mut minor_data = Vec::with_capacity(n - 1);

        for i in 0..n {
            if i == row {
                continue;
            }
            let mut new_row = Vec::with_capacity(n - 1);
            for j in 0..n {
                if j == col {
                    continue;
                }
                new_row.push(self.data[i][j]);
            }
            minor_data.push(new_row);
        }

        Matrix::new(minor_data)
    }

    /// Computes the inverse of this matrix, if it exists.
    ///
    /// The inverse matrix A⁻¹ of a square matrix A is the unique matrix such that:
    /// AA⁻¹ = A⁻¹A = I
    /// where I is the identity matrix.
    ///
    /// Uses the classical adjugate method:
    /// 1. Compute determinant
    /// 2. For each element (i,j), compute cofactor (using minor determinants)
    /// 3. Transpose the cofactor matrix to get adjugate
    /// 4. Divide adjugate by determinant
    ///
    /// # Complexity
    /// - Time: O(n³) where n is matrix dimension
    /// - Space: O(n²) for storing results
    ///
    /// # Return Value
    /// - Ok(Matrix) containing inverse if matrix is invertible
    /// - Err(MatrixError::Singular) if matrix is not invertible
    /// - Err(MatrixError::NotSquare) if matrix is not square
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [4.0, 7.0],
    ///     [2.0, 6.0]
    /// ]);
    /// let inv = m.inverse().unwrap();
    /// ```
    pub fn inverse(&self) -> Result<Matrix<K>, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }

        let n = self.rows();
        let det = self.determinant();

        // Check if matrix is invertible using tolerance
        if det.to_f32().abs() < Self::TOLERANCE {
            return Err(MatrixError::Singular);
        }

        let mut inverse_data = vec![vec![K::zero(); n]; n];

        // Calculate each element of the adjugate matrix, then divide by determinant
        for i in 0..n {
            for j in 0..n {
                // Get minor determinant by excluding row i and column j
                let minor_det = self.get_minor(i, j).determinant();

                // Calculate cofactor: (-1)^(i+j) * det(minor)
                let cofactor = if (i + j) % 2 == 0 {
                    minor_det
                } else {
                    -minor_det
                };

                // Transpose while building (hence j,i instead of i,j)
                // and divide by determinant
                inverse_data[j][i] = cofactor / det;
            }
        }

        Ok(Matrix::new(inverse_data))
    }

    /// Computes the rank of the matrix.
    ///
    /// The rank of a matrix is the number of linearly independent rows
    /// in its row-echelon form. Rows are considered independent if they
    /// contain non-zero values.
    ///
    /// # Complexity
    /// - Time: O(n³) due to the row-echelon computation
    /// - Space: O(n²) for intermediate storage
    ///
    /// # Example
    /// ```
    /// use matrix::Matrix;
    ///
    /// let m = Matrix::from([
    ///     [1.0, 2.0, 0.0, 0.0],
    ///     [2.0, 4.0, 0.0, 0.0],
    ///     [-1.0, 2.0, 1.0, 1.0],
    /// ]);
    /// assert_eq!(m.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        let rref = self.row_echelon();
        rref.data
            .iter()
            .filter(|row| row.iter().any(|&val| val.to_f32().abs() > Self::TOLERANCE))
            .count()
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

// Implementations for vector assign multiplication, needed for lerp
impl<K: Scalar> MulAssign<K> for Matrix<K> {
    fn mul_assign(&mut self, rhs: K) {
        self.scl(rhs);
    }
}

// Implementations for vector assign addition, needed for lerp
impl<K: Scalar> AddAssign for Matrix<K> {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs);
    }
}

// Implementations for vector assign subtraction, needed for lerp
impl<K: Scalar> SubAssign for Matrix<K> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(&rhs);
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

    mod operation_tests {
        use super::*;

        #[test]
        fn test_matrix_add() {
            let mut m1 = create_test_matrix();
            let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
            m1.add(&m2);
            assert_eq!(m1.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
        }

        #[test]
        fn test_matrix_add_assign() {
            let mut m1 = create_test_matrix();
            let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
            m1 += m2;
            assert_eq!(m1.data, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
        }

        #[test]
        #[should_panic(expected = "Addition requires matrices with the same shape.")]
        fn test_matrix_add_panic() {
            let mut m1 = create_test_matrix();
            let m2 = create_test_rect_matrix();
            m1.add(&m2);
        }

        mod evaluation_add_tests {
            use super::*;

            #[test]
            fn test_add_zero() {
                let mut m1 = Matrix::from([[0, 0], [0, 0]]);
                let m2 = Matrix::from([[0, 0], [0, 0]]);
                m1.add(&m2);
                assert_eq!(m1.data, vec![vec![0, 0], vec![0, 0]]);
            }

            #[test]
            fn test_add_identity() {
                let mut m1 = Matrix::from([[1, 0], [0, 1]]);
                let m2 = Matrix::from([[0, 0], [0, 0]]);
                m1.add(&m2);
                assert_eq!(m1.data, vec![vec![1, 0], vec![0, 1]]);
            }

            #[test]
            fn test_add_same() {
                let mut m1 = Matrix::from([[1, 1], [1, 1]]);
                let m2 = Matrix::from([[1, 1], [1, 1]]);
                m1.add(&m2);
                assert_eq!(m1.data, vec![vec![2, 2], vec![2, 2]]);
            }

            #[test]
            fn test_add_42() {
                let mut m1 = Matrix::from([[21, 21], [21, 21]]);
                let m2 = Matrix::from([[21, 21], [21, 21]]);
                m1.add(&m2);
                assert_eq!(m1.data, vec![vec![42, 42], vec![42, 42]]);
            }
        }

        #[test]
        fn test_matrix_sub() {
            let mut m1 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
            let m2 = create_test_matrix();
            m1.sub(&m2);
            assert_eq!(m1.data, vec![vec![4.0, 4.0], vec![4.0, 4.0]]);
        }

        #[test]
        fn test_matrix_sub_assign() {
            let mut m1 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
            let m2 = create_test_matrix();
            m1 -= m2;
            assert_eq!(m1.data, vec![vec![4.0, 4.0], vec![4.0, 4.0]]);
        }

        #[test]
        #[should_panic(expected = "Subtraction requires matrices with the same shape.")]
        fn test_matrix_sub_panic() {
            let mut m1 = create_test_matrix();
            let m2 = create_test_rect_matrix();
            m1.sub(&m2);
        }

        mod evaluation_sub_tests {
            use super::*;

            #[test]
            fn test_sub_zero() {
                let mut m1 = Matrix::from([[0, 0], [0, 0]]);
                let m2 = Matrix::from([[0, 0], [0, 0]]);
                m1.sub(&m2);
                assert_eq!(m1.data, vec![vec![0, 0], vec![0, 0]]);
            }

            #[test]
            fn test_sub_identity() {
                let mut m1 = Matrix::from([[1, 0], [0, 1]]);
                let m2 = Matrix::from([[0, 0], [0, 0]]);
                m1.sub(&m2);
                assert_eq!(m1.data, vec![vec![1, 0], vec![0, 1]]);
            }

            #[test]
            fn test_sub_same() {
                let mut m1 = Matrix::from([[1, 1], [1, 1]]);
                let m2 = Matrix::from([[1, 1], [1, 1]]);
                m1.sub(&m2);
                assert_eq!(m1.data, vec![vec![0, 0], vec![0, 0]]);
            }

            #[test]
            fn test_sub_42() {
                let mut m1 = Matrix::from([[21, 21], [21, 21]]);
                let m2 = Matrix::from([[21, 21], [21, 21]]);
                m1.sub(&m2);
                assert_eq!(m1.data, vec![vec![0, 0], vec![0, 0]]);
            }
        }

        #[test]
        fn test_matrix_scl() {
            let mut m = create_test_matrix();
            m.scl(2.0);
            assert_eq!(m.data, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
        }

        #[test]
        fn test_matrix_mul_assign() {
            let mut m = create_test_matrix();
            m *= 2.0;
            assert_eq!(m.data, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
        }

        #[test]
        fn test_matrix_zero_scale() {
            let mut m = create_test_matrix();
            m.scl(0.0);
            assert_eq!(m.data, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        }

        mod evaluation_scl_tests {
            use super::*;

            #[test]
            fn test_scl_zero() {
                let mut m = Matrix::from([[0, 0], [0, 0]]);
                m.scl(0);
                assert_eq!(m.data, vec![vec![0, 0], vec![0, 0]]);
            }

            #[test]
            fn test_scl_identity() {
                let mut m = Matrix::from([[1, 0], [0, 1]]);
                m.scl(1);
                assert_eq!(m.data, vec![vec![1, 0], vec![0, 1]]);
            }

            #[test]
            fn test_scl_2() {
                let mut m = Matrix::from([[1, 2], [3, 4]]);
                m.scl(2);
                assert_eq!(m.data, vec![vec![2, 4], vec![6, 8]]);
            }

            #[test]
            fn test_scl_half() {
                let mut m = Matrix::from([[21.0, 21.0], [21.0, 21.0]]);
                m.scl(0.5);
                assert_eq!(m.data, vec![vec![10.5, 10.5], vec![10.5, 10.5]]);
            }
        }
    }

    mod matrix_multiplication_tests {
        use super::*;

        #[test]
        fn test_matrix_vector_multiplication() {
            let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            let v = Vector::from([1.0, 2.0]);
            let result = m.mul_vec(&v);
            assert_eq!(result.data, vec![5.0, 11.0]);
        }

        #[test]
        fn test_matrix_vector_identity() {
            let m = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
            let v = Vector::from([1.0, 2.0]);
            let result = m.mul_vec(&v);
            assert_eq!(result.data, v.data);
        }

        #[test]
        fn test_matrix_vector_zero() {
            let m = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
            let v = Vector::from([1.0, 2.0]);
            let result = m.mul_vec(&v);
            assert_eq!(result.data, vec![0.0, 0.0]);
        }

        #[test]
        #[should_panic(expected = "Matrix columns must match vector size")]
        fn test_matrix_vector_wrong_dimensions() {
            let m = Matrix::from([[1.0, 2.0]]);
            let v = Vector::from([1.0, 2.0, 3.0]);
            m.mul_vec(&v);
        }

        #[test]
        fn test_matrix_matrix_multiplication() {
            let m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
            let result = m1.mul_mat(&m2);
            assert_eq!(result.data, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
        }

        #[test]
        fn test_matrix_matrix_identity() {
            let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            let i = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
            let result = m.mul_mat(&i);
            assert_eq!(result.data, m.data);
        }

        #[test]
        fn test_matrix_matrix_zero() {
            let m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            let m2 = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
            let result = m1.mul_mat(&m2);
            assert_eq!(result.data, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        }

        #[test]
        #[should_panic(expected = "First matrix columns must match second matrix rows")]
        fn test_matrix_matrix_wrong_dimensions() {
            let m1 = Matrix::from([[1.0, 2.0]]);
            let m2 = Matrix::from([[1.0], [2.0], [3.0]]);
            m1.mul_mat(&m2);
        }
    }

    mod trace_tests {
        use super::*;

        #[test]
        fn test_trace_2x2() {
            let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            assert_eq!(m.trace(), 5.0);
        }

        #[test]
        fn test_trace_identity() {
            let m = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
            assert_eq!(m.trace(), 3.0);
        }

        #[test]
        fn test_trace_zero() {
            let m = Matrix::from([[0.0, 1.0], [1.0, 0.0]]);
            assert_eq!(m.trace(), 0.0);
        }

        #[test]
        #[should_panic(expected = "Trace only defined for square matrices")]
        fn test_trace_non_square() {
            let m = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            m.trace();
        }
    }

    mod transpose_tests {
        use super::*;

        #[test]
        fn test_transpose_square() {
            let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            let m_t = m.transpose();
            assert_eq!(m_t.data, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
        }

        #[test]
        fn test_transpose_rectangular() {
            let m = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            let m_t = m.transpose();
            assert_eq!(
                m_t.data,
                vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
            );
        }

        #[test]
        fn test_transpose_identity() {
            let m = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
            let m_t = m.transpose();
            assert_eq!(m_t.data, m.data);
        }

        #[test]
        fn test_transpose_twice() {
            let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            let m_tt = m.transpose().transpose();
            assert_eq!(m_tt.data, m.data);
        }
    }

    mod row_echelon_tests {
        use super::*;

        #[test]
        #[should_panic(expected = "Cannot compute RREF of an empty matrix.")]
        fn test_row_echelon_empty_matrix() {
            let m: Matrix<f32> = Matrix::new(vec![]);
            m.row_echelon();
        }

        #[test]
        #[should_panic(expected = "Matrix rows must have the same number of columns.")]
        fn test_row_echelon_inconsistent_rows() {
            // Create matrix with inconsistent row lengths
            let mut data = vec![vec![1.0, 2.0, 3.0]];
            data.push(vec![4.0, 5.0]); // Second row is shorter
            let m = Matrix::new(data);
            m.row_echelon();
        }

        #[test]
        fn test_row_echelon_simple() {
            let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            let rref = m.row_echelon();

            // Should reduce to identity matrix
            let result_data = rref.data;
            assert_eq!(result_data[0][0], 1.0);
            assert_eq!(result_data[0][1], 0.0);
            assert_eq!(result_data[1][0], 0.0);
            assert_eq!(result_data[1][1], 1.0);
        }

        #[test]
        fn test_tolerance_significance() {
            // Create two nearly linearly dependent rows, but mathematically independent
            // First row is [1, 1]
            // Second row is [1, 1 + 1e-8] (difference > 1e-10)
            let m = Matrix::from([[1.0, 1.0], [1.0, 1.0 + 1e-8]]);

            let rref = m.row_echelon();

            // With our strict tolerance (1e-10), this small difference should be detected
            assert_eq!(rref.data[0][0], 1.0); // First row normalized
            assert_eq!(rref.data[0][1], 1.0); // Should be 1, not 0
            assert_eq!(rref.data[1][0], 0.0); // Should be eliminated
            assert!(rref.data[1][1] - 1e-8 < 1e-10); // Should have the small difference
        }

        #[test]
        fn test_actual_linear_dependence() {
            let m = Matrix::from([
                [1.0, 1.0],
                [1.0, 1.0], // Exactly same row
            ]);

            let rref = m.row_echelon();

            // First row should be [1, 1], second row should be zeroed
            assert_eq!(rref.data[0][0], 1.0);
            assert_eq!(rref.data[0][1], 1.0);
            assert_eq!(rref.data[1][0], 0.0);
            assert_eq!(rref.data[1][1], 0.0);
        }

        #[test]
        fn test_below_tolerance_zeroing() {
            let m = Matrix::from([
                [1.0, 1.0],
                [1.0, 1.0 + 1e-11], // Difference smaller than tolerance
            ]);

            let rref = m.row_echelon();

            // Should treat rows as dependent since difference is below tolerance
            assert_eq!(rref.data[0][0], 1.0);
            assert_eq!(rref.data[0][1], 1.0);
            assert_eq!(rref.data[1][0], 0.0);
            assert_eq!(rref.data[1][1], 0.0);
        }

        #[test]
        fn test_row_echelon_subject_example() {
            let m = Matrix::from([
                [8.0, 5.0, -2.0, 4.0, 28.0],
                [4.0, 2.5, 20.0, 4.0, -4.0],
                [8.0, 5.0, 1.0, 4.0, 17.0],
            ]);
            let rref = m.row_echelon();

            // Check first row - [1.0, 0.625, 0.0, 0.0, -12.1666667]
            assert_eq!(rref.data[0][0], 1.0);
            assert_eq!(rref.data[0][1], 0.625);
            assert_eq!(rref.data[0][2], 0.0);
            assert_eq!(rref.data[0][3], 0.0);
            assert!((rref.data[0][4] - (-12.1666667)).to_f32().abs() < 1e-6);

            // Check second row - [0.0, 0.0, 1.0, 0.0, -3.6666667]
            assert_eq!(rref.data[1][0], 0.0);
            assert_eq!(rref.data[1][1], 0.0);
            assert_eq!(rref.data[1][2], 1.0);
            assert_eq!(rref.data[1][3], 0.0);
            assert_eq!(rref.data[1][4], -3.6666667);

            // Check third row - [0.0, 0.0, 0.0, 1.0, 29.5]
            assert_eq!(rref.data[2][0], 0.0);
            assert_eq!(rref.data[2][1], 0.0);
            assert_eq!(rref.data[2][2], 0.0);
            assert_eq!(rref.data[2][3], 1.0);
            assert_eq!(rref.data[2][4], 29.5);
        }

        #[test]
        fn test_row_echelon_zero_matrix() {
            let m = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
            let rref = m.row_echelon();

            // Zero matrix should remain zero
            assert_eq!(rref.data, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        }

        #[test]
        fn test_row_echelon_identity() {
            let m = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
            let rref = m.row_echelon();

            // Identity matrix should remain unchanged
            assert_eq!(rref.data, m.data);
        }

        #[test]
        fn test_row_echelon_dependent_rows() {
            let m = Matrix::from([[1.0, 2.0], [2.0, 4.0]]);
            let rref = m.row_echelon();

            // Second row should reduce to zero
            assert_eq!(rref.data[0][0], 1.0);
            assert_eq!(rref.data[0][1], 2.0);
            assert_eq!(rref.data[1][0], 0.0);
            assert_eq!(rref.data[1][1], 0.0);
        }
    }
    mod determinant_tests {
        use super::*;

        #[test]
        fn test_det_1x1() {
            let m = Matrix::from([[2.0]]);
            assert_eq!(m.determinant(), 2.0);
        }

        #[test]
        fn test_det_2x2() {
            // Test identity matrix
            let m = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
            assert_eq!(m.determinant(), 1.0);

            // Test known determinant
            let m = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
            assert_eq!(m.determinant(), -2.0);

            // Test singular matrix
            let m = Matrix::from([[1.0, 2.0], [2.0, 4.0]]);
            assert_eq!(m.determinant(), 0.0);
        }

        #[test]
        fn test_det_3x3() {
            // Test identity matrix
            let m = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
            assert_eq!(m.determinant(), 1.0);

            // Test example from subject
            let m = Matrix::from([[8.0, 5.0, -2.0], [4.0, 7.0, 20.0], [7.0, 6.0, 1.0]]);
            assert_eq!(m.determinant(), -174.0);

            // Test singular matrix
            let m = Matrix::from([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]);
            assert_eq!(m.determinant(), 0.0);
        }

        #[test]
        fn test_det_4x4() {
            // Test identity matrix
            let m = Matrix::from([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            assert_eq!(m.determinant(), 1.0);

            // Test example from subject
            let m = Matrix::from([
                [8.0, 5.0, -2.0, 4.0],
                [4.0, 2.5, 20.0, 4.0],
                [8.0, 5.0, 1.0, 4.0],
                [28.0, -4.0, 17.0, 1.0],
            ]);
            assert_eq!(m.determinant(), 1032.0);
        }

        #[test]
        #[should_panic(expected = "Determinant only defined for square matrices")]
        fn test_det_non_square() {
            let m = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            m.determinant();
        }

        #[test]
        #[should_panic(expected = "Determinant implementation limited to 4x4 matrices")]
        fn test_det_too_large() {
            let m = Matrix::from([
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]);
            m.determinant();
        }
    }

    mod inverse_tests {
        use super::*;

        #[test]
        fn test_inverse_identity() {
            let m = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
            let inv = m.inverse().unwrap();
            assert_eq!(inv.data, m.data);
        }

        #[test]
        fn test_inverse_2x2() {
            let m = Matrix::from([[4.0, 7.0], [2.0, 6.0]]);
            let inv = m.inverse().unwrap();
            let expected = Matrix::from([[0.6, -0.7], [-0.2, 0.4]]);

            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (inv.data[i][j] - expected.data[i][j]).to_f32().abs()
                            < Matrix::<f32>::TOLERANCE
                    );
                }
            }
        }

        #[test]
        fn test_inverse_singular() {
            let m = Matrix::from([[1.0, 2.0], [2.0, 4.0]]);
            assert_eq!(m.inverse(), Err(MatrixError::Singular));
        }

        #[test]
        fn test_inverse_not_square() {
            let m = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            assert_eq!(m.inverse(), Err(MatrixError::NotSquare));
        }

        #[test]
        fn test_inverse_multiplication() {
            let a = Matrix::from([[4.0, 3.0], [3.0, 2.0]]);
            let a_inv = a.inverse().unwrap();
            let prod = a.mul_mat(&a_inv);

            let identity = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);

            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (prod.data[i][j] - identity.data[i][j]).to_f32().abs()
                            < Matrix::<f32>::TOLERANCE
                    );
                }
            }
        }

        #[test]
        fn test_inverse_from_subject() {
            let u = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
            let inv = u.inverse().unwrap();
            let expected = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
            assert_eq!(inv.data, expected.data);

            let u = Matrix::from([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]);
            let inv = u.inverse().unwrap();
            let expected = Matrix::from([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]);
            assert_eq!(inv.data, expected.data);

            let u = Matrix::from([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]);
            let inv = u.inverse().unwrap();
            let expected = Matrix::from([
                [0.649425287, 0.097701149, -0.655172414],
                [-0.781609195, -0.126436782, 0.965517241],
                [0.143678161, 0.074712644, -0.206896552],
            ]);
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (inv.data[i][j] - expected.data[i][j]).to_f32().abs()
                            < Matrix::<f32>::TOLERANCE
                    );
                }
            }
        }
    }

    mod rank_tests {
        use super::*;

        #[test]
        fn test_rank_full_rank() {
            let m = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
            assert_eq!(m.rank(), 3);
        }

        #[test]
        fn test_rank_partial_rank() {
            let m = Matrix::from([
                [1.0, 2.0, 0.0, 0.0],
                [2.0, 4.0, 0.0, 0.0],
                [-1.0, 2.0, 1.0, 1.0],
            ]);
            assert_eq!(m.rank(), 2);
        }

        #[test]
        fn test_rank_zero_matrix() {
            let m = Matrix::from([[0.0, 0.0], [0.0, 0.0]]);
            assert_eq!(m.rank(), 0);
        }

        #[test]
        fn test_rank_non_square_matrix() {
            let m = Matrix::from([
                [8.0, 5.0, -2.0],
                [4.0, 7.0, 20.0],
                [7.0, 6.0, 1.0],
                [21.0, 18.0, 7.0],
            ]);
            assert_eq!(m.rank(), 3);
        }
    }
}
