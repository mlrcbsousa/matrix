//! A linear algebra library for matrix operations
//!
//! This library provides implementations for vector and matrix operations
//! including addition, multiplication, determinants, and more.

/// Returns a friendly greeting for matrix enthusiasts
///
/// # Examples
///
/// ```
/// use matrix::say_hello;
/// println!("{}", say_hello());  // Prints: Hello, Matrix World!
/// ```
pub fn say_hello() -> &'static str {
    "Hello, Matrix World!"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeting() {
        assert_eq!(say_hello(), "Hello, Matrix World!");
    }
}