//! Implements the computation of a 3D projection matrix to map 3D points to screen space.

use crate::Matrix;

/// Computes a projection matrix for 3D rendering.
///
/// In 3D graphics, a perspective projection matrix transforms 3D points in camera space to 2D points
/// on the screen, applying a perspective effect where objects farther from the camera appear smaller.
///
/// The symmetric frustum is defined by the field of view (`fov`), aspect ratio (`ratio`), and near and far
/// clipping planes. These parameters define the shape and scale of the view frustum.
///
/// # Parameters
/// * `fov`: Field of View (in radians).
///   - Specifies the vertical angle of the viewing cone. A wider FOV results in a broader perspective.
/// * `ratio`: Aspect ratio (width / height).
///   - Adjusts horizontal scaling to match the screen dimensions.
/// * `near`: Near clipping plane distance.
///   - The closest depth where objects remain visible. Must be positive and less than `far`.
/// * `far`: Far clipping plane distance.
///   - The farthest depth where objects remain visible. Must be greater than `near`.
///
/// # Returns
/// A 4x4 perspective projection matrix (`Matrix<f32>`) used to transform 3D points into screen space.
///
/// ## Explanation of Matrix Elements
/// 1. **Scaling Factors**:
///    - The x and y scaling terms adjust based on the field of view and aspect ratio.
/// 2. **Depth Mapping**:
///    - Maps the depth range `[near, far]` to normalized device coordinates `[-1, 1]`.
/// 3. **Perspective Division**:
///    - Ensures proper perspective transformations for realistic rendering.
///
/// # Panics
/// Panics if `near` or `far` are non-positive or if `near >= far`, as this results in an invalid frustum.
///
/// # Example
/// ```rust
/// use matrix::projection;
///
/// let proj_matrix = projection(90.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
/// println!("{:?}", proj_matrix);
/// ```
pub fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix<f32> {
    if near <= 0.0 || far <= 0.0 || near >= far {
        panic!("Invalid near and far clipping planes.");
    }

    // Calculate tangent of half the field of view for vertical scaling.
    let tan_half_fov = (fov / 2.0).tan();
    let depth = far - near;

    let t = near * tan_half_fov;
    let b = -t;
    let r = t * ratio;
    let l = -r;

    // Populate the matrix.
    let mut data = vec![vec![0.0; 4], vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]];

    data[0][0] = 2.0 * near / (r - l); // Scaling x
    data[1][1] = 2.0 * near / (t - b); // Scaling y
    data[2][0] = (r + l) / (r - l); // Horizontal offset
    data[2][1] = (t + b) / (t - b); // Vertical offset
    data[2][2] = -(far + near) / depth; // Depth normalization
    data[2][3] = -(2.0 * far * near) / depth; // Depth mapping
    data[3][2] = -1.0; // Perspective division adjustment

    Matrix::new(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_valid() {
        let proj = projection(90.0_f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        assert!(proj.data[0][0] > 0.0);
        assert!(proj.data[1][1] > 0.0);
        assert!(proj.data[2][2] < 0.0);
        assert!(proj.data[3][2] < 0.0);
    }

    #[test]
    #[should_panic(expected = "Invalid near and far clipping planes.")]
    fn test_projection_invalid_planes() {
        projection(90.0_f32.to_radians(), 16.0 / 9.0, -0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "Invalid near and far clipping planes.")]
    fn test_projection_near_greater_than_far() {
        projection(90.0_f32.to_radians(), 16.0 / 9.0, 100.0, 0.1);
    }
}
