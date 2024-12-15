use matrix::{Matrix, Scalar, Vector};

/// Demonstrates the features of the Scalar trait and its implementations
fn type_scalar() {
    println!("\nScalar Type Demo");
    println!("-----------------");

    // Demonstrate identity elements
    let zero = f32::zero();
    let one = f32::one();
    println!("Identity elements:");
    println!("zero(): {}", zero);
    println!("one(): {}", one);

    // Demonstrate arithmetic operations
    let a: f32 = 2.0;
    let b: f32 = 3.0;
    println!("\nArithmetic operations:");
    println!("a = {}, b = {}", a, b);
    println!("a + b = {}", a + b);
    println!("a - b = {}", a - b);
    println!("a * b = {}", a * b);
    println!("a / b = {}", a / b);
}

/// Demonstrates the features of the Vector type
fn type_vector() {
    println!("\nVector Type Demo");
    println!("-----------------");

    // Create vectors using different constructors
    println!("Vector construction:");
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::from([4.0, 5.0, 6.0]);

    println!("\nVector created with new():");
    v1.print();
    println!("\nVector created with from():");
    v2.print();

    // Demonstrate size operations
    println!("\nVector sizes:");
    println!("v1 size: {}", v1.size());
    println!("v2 size: {}", v2.size());
}

/// Demonstrates the features of the Matrix type
fn type_matrix() {
    println!("\nMatrix Type Demo");
    println!("-----------------");

    // Create matrices using different constructors
    println!("Matrix construction:");
    let m1 = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);

    println!("\nMatrix created with new():");
    m1.print();
    println!("\nMatrix created with from():");
    m2.print();

    // Demonstrate dimension operations
    println!("\nMatrix dimensions:");
    println!("m1 shape: {:?}", m1.shape());
    println!("m1 rows: {}", m1.rows());
    println!("m1 cols: {}", m1.cols());
    println!("m1 is square: {}", m1.is_square());

    // Demonstrate with rectangular matrix
    let rect_matrix = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    println!("\nRectangular matrix:");
    rect_matrix.print();
    println!("shape: {:?}", rect_matrix.shape());
    println!("is square: {}", rect_matrix.is_square());
}

fn exercise_00() {
    println!();
    println!("Exercise 00 - Add, Subtract and Scale");
}

fn main() {
    println!("Matrix Library Demo");
    println!("==================");

    type_scalar();
    type_vector();
    type_matrix();
    exercise_00();
}
