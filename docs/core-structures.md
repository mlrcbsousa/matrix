# Core Structures

## Types and Interfaces

### 1. **Field Type**

The `Field` type is designed to be a union of numeric types that can be used within matrix and vector operations. This is the base type that the generic `K` in `Matrix<K>` and `Vector<K>` extends.

```typescript
export type Field = number | bigint | Arithmetic;
```

- `Field` is flexible and allows working with:
  - `number` types
  - `bigint` types
  - Custom types that implement the `Arithmetic` interface (e.g., `ComplexNumber` or `SomeField` in the [test examples](./testing#mocks)).

### 2. **Arithmetic Interface**

The `Arithmetic` interface defines the blueprint for any object that supports addition (`add`), subtraction (`sub`), and multiplication (`mul`) operations. This interface allows for custom numeric types to be used in matrices and vectors.

```typescript
export interface Arithmetic {
  add(other: Arithmetic): Arithmetic;
  sub(other: Arithmetic): Arithmetic;
  mul(other: Arithmetic): Arithmetic;
}
```

- Methods:
  - `add(other: Arithmetic): Arithmetic`: Performs addition.
  - `sub(other: Arithmetic): Arithmetic`: Performs subtraction.
  - `mul(other: Arithmetic): Arithmetic`: Performs multiplication.

### 3. **Display Interface**

The `Display` interface is implemented by classes that need to provide a custom string representation of their data.

```typescript
export interface Display {
  toString(): string;
}
```

- Method:
  - `toString(): string`: Returns a string representation of the class instance.

## Matrix Class

### Description
The `Matrix` class is a generic type that supports basic matrix operations. It can be initialized with any type `K` that extends the `Field` type (i.e., numbers, bigints, or custom arithmetic types).

### Constructor

```typescript
constructor(public values: K[][])
```

- `values`: A 2D array of type `K`. Each row of the matrix must have the same length.
- The constructor validates that the matrix has at least one row and that all rows have equal length.

### Methods

#### `size()`

```typescript
size(): number[]
```

- Returns the dimensions of the matrix as an array `[number of rows, number of columns]`.

#### `isSquare()`

```typescript
isSquare(): boolean
```

- Returns `true` if the matrix is square (i.e., has equal number of rows and columns).

#### `toString()`

```typescript
toString(): string
```

- Returns a string representation of the matrix, printing each row on a new line.

## Vector Class

### Description

The `Vector` class represents a one-dimensional vector and supports basic vector operations. Like `Matrix`, it is also generic and can work with numbers, bigints, or custom fields.

### Constructor

```typescript
constructor(public values: K[])
```

- `values`: An array of type `K`.

### Methods

#### `size()`

```typescript
size(): number
```

- Returns the size (length) of the vector.

#### `toString()`

```typescript
toString(): string
```

- Returns a string representation of the vector in the format `Vector { <value1>, <value2>, ... }`.
