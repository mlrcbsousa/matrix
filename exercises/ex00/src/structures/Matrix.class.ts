// TODO: A function to reshape a vector into a matrix, and vice-versa.

interface Structure<K> {
  values: K[] | K[][];
  size(): number | number[];
  toString(): string;
}

interface Arithmetic<T extends Structure<K>, K> {
  add(other: T): T;
  sub(other: T): T;
  scl(scalar: K): T;
}

export class Vector<K> implements Arithmetic<Vector<K>, K> {
  constructor(public values: K[]) {}
  // TODO: validate values

  size(): number {
    return this.values.length;
  }

  toString(): string {
    return `Vector { ${this.values.join(', ')} }`;
  }

  add(other: Vector<K>): Vector<K> {
    throw new Error("Method not implemented.");
  }

  sub(other: Vector<K>): Vector<K> {
    throw new Error("Method not implemented.");
  }

  scl(scalar: K): Vector<K> {
    throw new Error("Method not implemented.");
  }
}

export class Matrix<K> implements Arithmetic<Matrix<K>, K> {
  constructor(public values: K[][]) {}
  // TODO: validate every number[] same length otherwise size can be not valid

  size(): number[] {
    return [this.values.length, this.values[0]!.length];
  }

  isSquare(): boolean {
    return this.values.length === this.values[0]!.length;
  }

  toString(): string {
    return `Matrix { ${this.values.map((row) => row.join(', ')).join('\n')} }`;
  }

  add(other: Matrix<K>): Matrix<K> {
    throw new Error("Method not implemented.");
  }

  sub(other: Matrix<K>): Matrix<K> {
    throw new Error("Method not implemented.");
  }

  scl(scalar: K): Matrix<K> {
    throw new Error("Method not implemented.");
  }
}