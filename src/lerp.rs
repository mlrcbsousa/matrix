use std::ops::{AddAssign, MulAssign, SubAssign};

/// Performs linear interpolation between two values.
///
/// Linear interpolation finds a value on the line between two endpoints.
/// - When t = 0, returns u
/// - When t = 1, returns v
/// - When t = 0.5, returns midpoint between u and v
///
/// Formula:
/// ```txt
/// lerp(u, v, t)   = u + t(v - u)
///                 = (1 - t)u + tv
/// ```
/// Both formulas are equivalent, but the second form has a nice geometric
/// interpretation: weighted average of vectors.
///
/// # Complexity
/// - Time: O(n) where n is the length - single pass over elements
/// - Space: O(n) - allocates a new result T
///
/// # Arguments
/// * `u` - Starting value
/// * `v` - Ending value
/// * `t` - Interpolation parameter between 0 and 1 // but works with any value
///
/// # Example
/// ```
/// use matrix::lerp;
///
/// let start = 0.0;
/// let end = 10.0;
/// let mid = lerp(start, end, 0.5);
/// println!("{}", mid); // 5.0
/// ```
pub fn lerp<T>(u: T, v: T, t: f32) -> T
where
    T: Clone + AddAssign + SubAssign + MulAssign<f32>,
{
    let mut a = u.clone();
    let mut b = v.clone();
    b -= u;
    b *= t;
    a += b;
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Matrix, Vector};

    #[test]
    fn test_lerp_f32() {
        assert_eq!(lerp(0.0, 1.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 1.0, 1.0), 1.0);
        assert_eq!(lerp(0.0, 1.0, 0.5), 0.5);
        assert_eq!(lerp(21.0, 42.0, 0.3), 27.3);
        assert_eq!(lerp(0.0, 10.0, 0.3), 3.0);
        assert_eq!(lerp(0.0, 10.0, -0.7), -7.0);
        assert_eq!(lerp(0.0, 10.0, 7.0), 70.0);
    }

    #[test]
    fn test_lerp_vector() {
        let v1 = create_test_vector_1();
        let v2 = create_test_vector_2();
        let result = lerp(v1, v2, 0.3);
        assert_eq!(result.data, vec![2.6, 1.3]);
    }

    #[test]
    fn test_lerp_vector_zero() {
        let v1 = create_test_vector_1();
        let v2 = create_test_vector_2();
        let result = lerp(v1, v2, 0.0);
        assert_eq!(result.data, vec![2.0, 1.0]);
    }

    #[test]
    fn test_lerp_vector_one() {
        let v1 = create_test_vector_1();
        let v2 = create_test_vector_2();
        let result = lerp(v1, v2, 1.0);
        assert_eq!(result.data, vec![4.0, 2.0]);
    }

    #[test]
    fn test_lerp_vector_f32() {
        let v1: Vector<f32> = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2: Vector<f32> = Vector::new(vec![3.0, 5.0, 6.0]);
        let t = 0.2;
        let result = lerp(v1, v2, t);
        assert_eq!(result.data, vec![1.4, 2.6, 3.6]);
    }

    #[test]
    fn test_lerp_matrix() {
        let m1 = create_test_matrix_1();
        let m2 = create_test_matrix_2();
        let result = lerp(m1, m2, 0.5);
        assert_eq!(result.data, vec![vec![11.0, 5.5], vec![16.5, 22.0]]);
    }

    #[test]
    fn test_lerp_matrix_zero() {
        let m1 = create_test_matrix_1();
        let m2 = create_test_matrix_2();
        let result = lerp(m1, m2, 0.0);
        assert_eq!(result.data, vec![vec![2.0, 1.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn test_lerp_matrix_one() {
        let m1 = create_test_matrix_1();
        let m2 = create_test_matrix_2();
        let result = lerp(m1, m2, 1.0);
        assert_eq!(result.data, vec![vec![20.0, 10.0], vec![30.0, 40.0]]);
    }

    // Helper functions to create test data
    fn create_test_vector_1() -> Vector<f32> {
        Vector::from([2.0, 1.0])
    }
    fn create_test_vector_2() -> Vector<f32> {
        Vector::from([4.0, 2.0])
    }
    fn create_test_matrix_1() -> Matrix<f32> {
        Matrix::from([[2.0, 1.0], [3.0, 4.0]])
    }
    fn create_test_matrix_2() -> Matrix<f32> {
        Matrix::from([[20.0, 10.0], [30.0, 40.0]])
    }
}
