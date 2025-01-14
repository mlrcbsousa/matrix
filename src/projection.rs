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

    // Calculate frustum dimensions
    let t = near * tan_half_fov; // top
    let b = -t; // bottom
    let r = t * ratio; // right
    let l = -r; // left

    // Build the matrix
    let mut data = vec![vec![0.0; 4]; 4];

    // Scale factors
    data[0][0] = 2.0 * near / (r - l); // X scale
    data[1][1] = 2.0 * near / (t - b); // Y scale

    // Perspective transformation
    data[2][0] = (r + l) / (r - l); // Horizontal offset
    data[2][1] = (t + b) / (t - b); // Vertical offset
    data[2][2] = -(far + near) / depth; // Depth normalization
    data[2][3] = -2.0 * far * near / depth; // Depth mapping

    data[3][2] = -1.0; // Perspective division

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

    mod evaluation_projection_tests {
        use super::*;

        #[test]
        fn test_projection_100degrees() {
            let fov = 100.0_f32.to_radians();
            let ratio = 1.0;
            let near = 0.1;
            let far = 100.0;
            let proj = projection(fov, ratio, near, far);

            // For wide FOV, expect smaller focal length (scale factors)
            assert!(proj.data[0][0].abs() < 1.5); // Scale factor should be relatively small for wide FOV
            assert!(proj.data[1][1].abs() < 1.5);
            // Check perspective division
            assert_eq!(proj.data[3][2], -1.0);
        }

        #[test]
        fn test_projection_70degrees() {
            let fov = 70.0_f32.to_radians();
            let ratio = 1.0;
            let near = 0.1;
            let far = 100.0;
            let proj = projection(fov, ratio, near, far);

            // For medium FOV, expect moderate focal length
            let scale = 1.0 / (fov/2.0).tan();
            assert!((proj.data[0][0] - scale).abs() < 1e-5);
            assert!((proj.data[1][1] - scale).abs() < 1e-5);
        }

        #[test]
        fn test_projection_40degrees() {
            let fov = 40.0_f32.to_radians();
            let ratio = 1.0;
            let near = 0.1;
            let far = 100.0;
            let proj = projection(fov, ratio, near, far);

            // For narrow FOV, expect larger focal length (zoomed in)
            assert!(proj.data[0][0].abs() > 2.0); // Scale factor should be larger for narrow FOV
            assert!(proj.data[1][1].abs() > 2.0);
        }

        // #[test]
        // fn test_projection_aspect_ratio() {
        //     let fov = 90.0_f32.to_radians();
        //     let ratio = 16.0/9.0;
        //     let near = 0.1;
        //     let far = 100.0;
        //     let proj = projection(fov, ratio, near, far);

        //     // Vertical and horizontal scale should differ by aspect ratio
        //     assert!((proj.data[0][0] / proj.data[1][1] - ratio).abs() < 1e-5);
        // }

        #[test]
        fn test_projection_near_far() {
            let fov = 90.0_f32.to_radians();
            let ratio = 1.0;
            let near = 1.0;
            let far = 10.0;
            let proj = projection(fov, ratio, near, far);

            // Check depth mapping coefficients
            let expected_c = -(far + near) / (far - near);
            let expected_d = -2.0 * far * near / (far - near);
            assert!((proj.data[2][2] - expected_c).abs() < 1e-5);
            assert!((proj.data[2][3] - expected_d).abs() < 1e-5);
        }

        #[test]
        #[should_panic(expected = "Invalid near and far clipping planes.")]
        fn test_projection_invalid_near() {
            projection(90.0_f32.to_radians(), 1.0, -0.1, 100.0);
        }

        #[test]
        #[should_panic(expected = "Invalid near and far clipping planes.")]
        fn test_projection_invalid_far() {
            projection(90.0_f32.to_radians(), 1.0, 0.1, -100.0);
        }

        #[test]
        #[should_panic(expected = "Invalid near and far clipping planes.")]
        fn test_projection_near_greater_than_far() {
            projection(90.0_f32.to_radians(), 1.0, 10.0, 1.0);
        }
    }
}
