# matrix

This repository contains the **matrix** project for [42 School](https://www.42network.org/), implementing a linear algebra library in Rust. The project focuses on implementing vector and matrix operations from scratch.

- [Documentation](https://mlrcbsousa.github.io/matrix/).

## Project Structure

```
matrix/
└── src/
    ├── lib.rs    # Library implementation
    └── main.rs   # Demo binary
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mlrcbsousa/matrix.git
cd matrix
```

## Usage

The project can be used both as a library and a demo binary.

### Run the Demo

```bash
cargo run
```

### Use as a Library

Add to your `Cargo.toml`:
```toml
[dependencies]
matrix = { git = "https://github.com/mlrcbsousa/matrix" }
```

## Development

### Testing

Run the test suite:
```bash
cargo test
```

### Code Quality

Format your code:
```bash
cargo fmt
```

Run the linter:
```bash
cargo clippy -- -D warnings
```

### Documentation

Generate and view documentation locally:
```bash
# Generate docs
cargo doc --no-deps

# Open in browser (on Unix systems)
open target/doc/matrix/index.html
```

## Documentation

The project documentation is automatically generated from code comments using `rustdoc`.

- Docs URL: [mlrcbsousa.github.io/matrix/](https://mlrcbsousa.github.io/matrix/)

### Deploying Documentation

Documentation is automatically deployed to GitHub Pages when:
- Changes are pushed to the **main** branch AND
- Changes affect:
  - Rust source files (`src/**/*.rs`)
  - `Cargo.toml`
  - Deploy configuration

Manual deployment can be triggered from the GitHub Actions tab using the `workflow_dispatch` event.

For deployment details check [`.github/workflows/deploy.yml`](/.github/workflows/deploy.yml)

## Git Hooks

This project uses git hooks for quality control. The following checks run automatically before each commit:

- Test suite (`cargo test`)
- Code formatting (`cargo fmt`)
- Linting (`cargo clippy`)
- Documentation generation (`cargo doc`)
- Demo build (`cargo build`)

These hooks are managed through `cargo-husky` and will be installed automatically when you build the project.
