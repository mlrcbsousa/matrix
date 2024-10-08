# matrix

This repository contains the **matrix** project for [42 School](https://www.42network.org/), focused on a series of exercises written in Rust and TypeScript. The exercises are located in the `exercises` folder and can be run individually.

- [Documentation](https://mlrcbsousa.github.io/matrix/).

## Project Structure

- `exercises/`: Contains all the exercises.
  - Exercises are written in **Rust** and **TypeScript**.
  - Each exercise is independent and can be executed individually.

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/mlrcbsousa/matrix.git
cd matrix
```

Navigate to the desired exercise folder:

```bash
cd exercises/ex00
```

Exercises go from `ex00` to `ex15`

### Install

Install TypeScript dependencies

```bash
npm install
```

### Running Rust exercises:

```bash
cargo run
```

Run the tests for that exercise

```
cargo test
```

### Running TypeScript exercises:

```bash
npm start
```

Run the tests for that exercise

```
npm test
```

## Documentation

The project documentation is built using [Vitepress](https://vitepress.dev/). It provides detailed explanations of the exercises and other useful information.

- Docs URL: [mlrcbsousa.github.io/matrix/](https://mlrcbsousa.github.io/matrix/)

### Install

Install TypeScript dependencies

```bash
npm install
```

### Building the Documentation

To start the documentation in development mode:

```bash
npm run docs:dev
```

To build the documentation for production:

```bash
npm run docs:build
```

To preview the production build locally:

```bash
npm run docs:preview
```

### Deploying the Documentation

The documentation is deployed to GitHub Pages. Ensure that your project is correctly set up to deploy to the GitHub repository `mlrcbsousa/matrix`.

Documentation is automatically deployed after pushes to the **main** branch and changes to the `docs/` folder or deploy config file.

For more on the deployment details check the file [`.github/workflows/deploy.yml`](/.github/workflows/deploy.yml)
