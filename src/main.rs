use matrix::{Matrix, Scalar, Vector};

/// Demonstrates the features of the Scalar trait and its implementations
fn type_scalar() {
    println!();
    println!("Scalar Type Demo");
    println!("----------------");

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
    println!();
    println!("Vector Type Demo");
    println!("----------------");

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
    println!();
    println!("Matrix Type Demo");
    println!("----------------");

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
    println!("-------------------------------------");

    let scalar = 2.0;

    let mut v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);

    println!("\nVector 1:");
    v1.print();
    println!("\nVector 2:");
    v2.print();

    v1.add(&v2);
    println!("\nVector 1 + Vector 2:");
    v1.print();

    v1.sub(&v2);
    println!("\nVector 1 - Vector 2:");
    v1.print();

    v1.scl(scalar);
    println!("\nVector 1 * {}: ", scalar);
    v1.print();

    v1.scl(0.0);
    println!("\nVector 1 * 0: ");
    v1.print();

    let mut m1 = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let m2 = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

    println!("\nMatrix 1:");
    m1.print();
    println!("\nMatrix 2:");
    m2.print();

    m1.add(&m2);
    println!("\nMatrix 1 + Matrix 2:");
    m1.print();

    m1.sub(&m2);
    println!("\nMatrix 1 - Matrix 2:");
    m1.print();

    m1.scl(scalar);
    println!("\nMatrix 1 * {}: ", scalar);
    m1.print();

    m1.scl(0.0);
    println!("\nMatrix 1 * 0: ");
    m1.print();
}

fn main() {
    println!("Matrix Library Demo");
    println!("===================");

    type_scalar();
    type_vector();
    type_matrix();
    exercise_00();
}
