use matrix::{angle_cos, cross_product, lerp, linear_combination, Matrix, Scalar, Vector};
use std::env;

fn print_usage() {
    println!();
    println!("Usage: matrix [exercise]");
    println!();
    println!("Available exercises:");
    println!("  types   - Scalar, Vector and Matrix types");
    println!("  ex00    - Add, Subtract and Scale");
    println!("  ex01    - Linear Combination");
    println!("  ex02    - Linear Interpolation");
    println!();
    println!("If no exercise specified, runs all exercises");
}

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

fn exercise_01() {
    println!();
    println!("Exercise 01 - Linear Combination");
    println!("--------------------------------");

    // Basic vectors
    let v1 = Vector::from([1.0, 2.0, 3.0]);
    let v2 = Vector::from([4.0, 5.0, 6.0]);
    let vectors = [v1.clone(), v2.clone()];
    let coefs = [2.0, -1.0];

    println!("\nInput vectors:");
    println!("Vector 1 (coef = {}):", coefs[0]);
    v1.print();
    println!("\nVector 2 (coef = {}):", coefs[1]);
    v2.print();

    let result = linear_combination(&vectors, &coefs);
    println!("\nLinear combination result:");
    result.print();

    // Standard basis example
    println!("\nStandard basis example:");
    let e1 = Vector::from([1.0, 0.0, 0.0]);
    let e2 = Vector::from([0.0, 1.0, 0.0]);
    let e3 = Vector::from([0.0, 0.0, 1.0]);

    println!("e1:");
    e1.print();
    println!("\ne2:");
    e2.print();
    println!("\ne3:");
    e3.print();

    let basis = [e1, e2, e3];
    let coords = [3.0, -1.0, 2.0];

    println!("\nCoordinates: {:?}", coords);
    let point = linear_combination(&basis, &coords);
    println!("\nPoint in R3:");
    point.print();
}

fn exercise_02() {
    println!();
    println!("Exercise 02 - Linear Interpolation");
    println!("----------------------------------");

    // Scalar interpolation
    let a = 0.0f32;
    let b = 10.0f32;
    println!("\nScalar interpolation:");
    println!("Start: {}", a);
    println!("End: {}", b);
    println!("t = 0.0: {}", lerp(a, b, 0.0));
    println!("t = 0.5: {}", lerp(a, b, 0.5));
    println!("t = 1.0: {}", lerp(a, b, 1.0));

    // Vector interpolation
    let v1 = Vector::from([1.0, 2.0]);
    let v2 = Vector::from([5.0, 6.0]);
    println!("\nVector interpolation:");
    println!("Start vector:");
    v1.print();
    println!("\nEnd vector:");
    v2.print();

    println!("\nt = 0.0:");
    lerp(v1.clone(), v2.clone(), 0.0).print();
    println!("\nt = 0.5:");
    lerp(v1.clone(), v2.clone(), 0.5).print();
    println!("\nt = 1.0:");
    lerp(v1.clone(), v2.clone(), 1.0).print();

    // Matrix interpolation
    let m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);
    println!("\nMatrix interpolation:");
    println!("Start matrix:");
    m1.print();
    println!("\nEnd matrix:");
    m2.print();

    println!("\nt = 0.0:");
    lerp(m1.clone(), m2.clone(), 0.0).print();
    println!("\nt = 0.5:");
    lerp(m1.clone(), m2.clone(), 0.5).print();
    println!("\nt = 1.0:");
    lerp(m1.clone(), m2.clone(), 1.0).print();
}

fn exercise_03() {
    println!();
    println!("Exercise 03 - Dot Product");
    println!("-------------------------");

    // Basic vector dot product
    let v1 = Vector::from([1.0, 2.0, 3.0]);
    let v2 = Vector::from([4.0, 5.0, 6.0]);
    println!("\nVector 1:");
    v1.print();
    println!("\nVector 2:");
    v2.print();
    println!("\nDot product: {}", v1.dot(&v2)); // 32.0

    // Geometric interpretation examples
    println!("\nGeometric meaning examples:");

    // Orthogonal vectors
    let v1 = Vector::from([1.0, 0.0]);
    let v2 = Vector::from([0.0, 1.0]);
    println!("\nOrthogonal vectors (perpendicular):");
    println!("v1:");
    v1.print();
    println!("v2:");
    v2.print();
    println!(
        "Dot product: {} (zero for perpendicular vectors)",
        v1.dot(&v2)
    );

    // Parallel vectors
    let v1 = Vector::from([2.0, 0.0]);
    let v2 = Vector::from([4.0, 0.0]);
    println!("\nParallel vectors (same direction):");
    println!("v1:");
    v1.print();
    println!("v2:");
    v2.print();
    println!("Dot product: {} (product of magnitudes)", v1.dot(&v2));

    // Opposite vectors
    let v1 = Vector::from([1.0, 0.0]);
    let v2 = Vector::from([-1.0, 0.0]);
    println!("\nOpposite vectors (opposite directions):");
    println!("v1:");
    v1.print();
    println!("v2:");
    v2.print();
    println!(
        "Dot product: {} (negative for opposing vectors)",
        v1.dot(&v2)
    );
}

fn exercise_04() {
    println!();
    println!("Exercise 04 - Norm");
    println!("------------------");

    // Create test vectors
    let v1 = Vector::from([1.0, 2.0, 2.0]);
    let v2 = Vector::from([-4.0, 0.0, 3.0]);

    println!("\nVector 1:");
    v1.print();
    println!("Manhattan norm (1-norm):   {}", v1.norm_1()); // 5.0
    println!("Euclidean norm (2-norm):   {}", v1.norm()); // 3.0
    println!("Supremum norm (inf-norm):  {}", v1.norm_inf()); // 2.0

    println!("\nVector 2:");
    v2.print();
    println!("Manhattan norm (1-norm):   {}", v2.norm_1()); // 7.0
    println!("Euclidean norm (2-norm):   {}", v2.norm()); // 5.0
    println!("Supremum norm (inf-norm):  {}", v2.norm_inf()); // 4.0

    // Demonstrate relationships between norms
    println!("\nRelationships between norms:");
    println!("For any vector v: ∥v∥∞ ≤ ∥v∥₂ ≤ ∥v∥₁");

    let v3 = Vector::from([3.0, -4.0]);
    println!("\nVector:");
    v3.print();
    println!(
        "∥v∥∞ = {} ≤ ∥v∥₂ = {} ≤ ∥v∥₁ = {}",
        v3.norm_inf(),
        v3.norm(),
        v3.norm_1()
    );
}

fn exercise_05() {
    println!();
    println!("Exercise 05 - Cosine");
    println!("--------------------");

    // Basic vectors
    let v1 = Vector::from([1.0, 0.0]);
    let v2 = Vector::from([0.0, 1.0]);

    println!("\nPerpendicular vectors (90°):");
    println!("v1:");
    v1.print();
    println!("v2:");
    v2.print();
    println!("cos(angle) = {}", angle_cos(&v1, &v2)); // 0.0

    // Parallel vectors
    let v3 = Vector::from([2.0, 0.0]);
    let v4 = Vector::from([4.0, 0.0]);
    println!("\nParallel vectors (0°):");
    println!("v3:");
    v3.print();
    println!("v4:");
    v4.print();
    println!("cos(angle) = {}", angle_cos(&v3, &v4)); // 1.0

    // Opposite vectors
    let v5 = Vector::from([1.0, 0.0]);
    let v6 = Vector::from([-1.0, 0.0]);
    println!("\nOpposite vectors (180°):");
    println!("v5:");
    v5.print();
    println!("v6:");
    v6.print();
    println!("cos(angle) = {}", angle_cos(&v5, &v6)); // -1.0

    // 45 degree angle
    let v7 = Vector::from([1.0, 0.0]);
    let v8 = Vector::from([1.0, 1.0]);
    println!("\n45° angle vectors:");
    println!("v7:");
    v7.print();
    println!("v8:");
    v8.print();
    println!("cos(angle) = {}", angle_cos(&v7, &v8)); // ~0.707
}

fn exercise_06() {
    println!();
    println!("Exercise 06 - Cross Product");
    println!("---------------------------");

    // Standard basis vectors
    let i = Vector::from([1.0, 0.0, 0.0]);
    let j = Vector::from([0.0, 1.0, 0.0]);
    let k = Vector::from([0.0, 0.0, 1.0]);

    println!("\nRight hand rule with basis vectors:");
    println!("i × j = k:");
    println!("i:");
    i.print();
    println!("j:");
    j.print();
    println!("Result:");
    cross_product(&i, &j).print();

    println!("\nj × k = i:");
    println!("j:");
    j.print();
    println!("k:");
    k.print();
    println!("Result:");
    cross_product(&j, &k).print();

    println!("\nk × i = j:");
    println!("k:");
    k.print();
    println!("i:");
    i.print();
    println!("Result:");
    cross_product(&k, &i).print();

    // Anticommutative property
    let u = Vector::from([2.0, 3.0, 4.0]);
    let v = Vector::from([5.0, 6.0, 7.0]);

    println!("\nAnticommutative property (u × v = -(v × u)):");
    println!("u:");
    u.print();
    println!("v:");
    v.print();
    println!("u × v:");
    cross_product(&u, &v).print();
    println!("v × u:");
    cross_product(&v, &u).print();

    // Practical example
    let u = Vector::from([4.0, 2.0, -3.0]);
    let v = Vector::from([-2.0, -5.0, 16.0]);
    println!("\nPractical example:");
    println!("u:");
    u.print();
    println!("v:");
    v.print();
    println!("u × v:");
    cross_product(&u, &v).print();
}

fn exercise_07() {
    println!();
    println!("Exercise 07 - Matrix Multiplication");
    println!("-----------------------------------");

    // Matrix-Vector multiplication
    println!("\nMatrix-Vector Multiplication:");
    let m = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
    let v = Vector::from([4.0, 2.0]);

    println!("Matrix (identity):");
    m.print();
    println!("Vector:");
    v.print();
    println!("Result (Mv):");
    m.mul_vec(&v).print();

    // Scaling matrix example
    println!("\nScaling matrix example:");
    let scale = Matrix::from([[2.0, 0.0], [0.0, 2.0]]);
    println!("Matrix (2x scaling):");
    scale.print();
    println!("Vector:");
    v.print();
    println!("Result (scales vector by 2):");
    scale.mul_vec(&v).print();

    // Matrix-Matrix multiplication
    println!("\nMatrix-Matrix Multiplication:");
    let m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    let m2 = Matrix::from([[5.0, 6.0], [7.0, 8.0]]);

    println!("Matrix 1:");
    m1.print();
    println!("Matrix 2:");
    m2.print();
    println!("Result (M1 × M2):");
    m1.mul_mat(&m2).print();

    // Special case: Identity
    println!("\nMultiplication with identity:");
    let identity = Matrix::from([[1.0, 0.0], [0.0, 1.0]]);
    println!("Matrix:");
    m1.print();
    println!("Identity matrix:");
    identity.print();
    println!("Result (M × I):");
    m1.mul_mat(&identity).print();
}

fn exercise_08() {
    println!();
    println!("Exercise 08 - Trace");
    println!("-------------------");

    // Basic trace example
    let m1 = Matrix::from([[1.0, 2.0], [3.0, 4.0]]);
    println!("\nMatrix:");
    m1.print();
    println!("Trace: {}", m1.trace()); // 5.0

    // Identity matrix trace equals dimension
    let identity = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    println!("\nIdentity matrix (3x3):");
    identity.print();
    println!("Trace: {} (equals matrix dimension)", identity.trace()); // 3.0

    // Diagonal matrix
    let diagonal = Matrix::from([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
    println!("\nDiagonal matrix:");
    diagonal.print();
    println!("Trace: {} (sum of diagonal elements)", diagonal.trace()); // 9.0

    // Symmetric matrix
    let symmetric = Matrix::from([[1.0, 2.0], [2.0, 1.0]]);
    println!("\nSymmetric matrix:");
    symmetric.print();
    println!("Trace: {}", symmetric.trace()); // 2.0
}

fn exercise_09() {
    println!();
    println!("Exercise 09 - Transpose");
    println!("-----------------------");

    // Square matrix example
    let m1 = Matrix::from([
        [1.0, 2.0],
        [3.0, 4.0]
    ]);
    println!("\nOriginal square matrix:");
    m1.print();
    println!("\nTransposed:");
    m1.transpose().print();

    // Rectangular matrix example
    let m2 = Matrix::from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]);
    println!("\nOriginal rectangular matrix:");
    m2.print();
    println!("\nTransposed:");
    m2.transpose().print();

    // Identity matrix example
    let identity = Matrix::from([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]);
    println!("\nIdentity matrix:");
    identity.print();
    println!("\nTransposed (should be identical):");
    identity.transpose().print();
}

fn main() {
    let args: Vec<String> = env::args().collect();

    println!("Matrix Library Demo");
    println!("===================");

    match args.get(1).map(|s| s.as_str()) {
        Some("types") => {
            type_scalar();
            type_vector();
            type_matrix();
        }
        Some("ex00") => exercise_00(),
        Some("ex01") => exercise_01(),
        Some("ex02") => exercise_02(),
        Some("ex03") => exercise_03(),
        Some("ex04") => exercise_04(),
        Some("ex05") => exercise_05(),
        Some("ex06") => exercise_06(),
        Some("ex07") => exercise_07(),
        Some("ex08") => exercise_08(),
        Some("ex09") => exercise_09(),
        // Some("ex10") => exercise_10(),
        // Some("ex11") => exercise_11(),
        // Some("ex12") => exercise_12(),
        // Some("ex13") => exercise_13(),
        // Some("ex14") => exercise_14(),
        // Some("ex15") => exercise_15(),
        Some("--help" | "-h") => print_usage(),
        None => {
            // Run all exercises
            type_scalar();
            type_vector();
            type_matrix();
            exercise_00();
            exercise_01();
            exercise_02();
            exercise_03();
            exercise_04();
            exercise_05();
            exercise_06();
            exercise_07();
            exercise_08();
            exercise_09();
            // exercise_10();
            // exercise_11();
            // exercise_12();
            // exercise_13();
            // exercise_14();
            // exercise_15();
        }
        Some(arg) => {
            println!("Unknown exercise: {}", arg);
            print_usage();
        }
    }
}
