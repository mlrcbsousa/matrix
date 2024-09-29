import { Field, Arithmetic, Display, Structure } from "./types.ts";

export default class Matrix<K extends Field> implements Structure<K, Matrix<K>>, Display {
  // TODO: A function to reshape a matrix into a vector, and vice-versa.

  constructor(public values: K[][]) {
    if (!values[0]) {
      throw new Error('Matrix must have at least 1 row');
    }

    const rowLength = values[0].length;
    if (!values.every((row) => row.length === rowLength)) {
      throw new Error('Every row must have the same length');
    }
  }

  size(): number[] {
    return [this.values.length, this.values[0]!.length];
  }

  isSquare(): boolean {
    return this.values.length === this.values[0]!.length;
  }

  toString(): string {
    return `Matrix {\n ${this.values.map((row) => row.join(', ')).join('\n ')}\n}`;
  }

  add(other: Matrix<K>): Matrix<K> {
    if (this.size().toString() !== other.size().toString()) {
      throw new Error('Matrices must have the same size');
    }

    this.values.forEach((row, i) => {
      row.forEach((value, j) => {
        const otherValue = other.values[i]![j];

        if (typeof value === 'number' && typeof otherValue === 'number') {
          this.values[i]![j] = (value + otherValue) as K;
        } else if (typeof value === 'bigint' && typeof otherValue === 'bigint') {
          this.values[i]![j] = (value + otherValue) as K;
        } else {
          (this.values[i]![j] as Arithmetic) = (value as Arithmetic).add((otherValue as Arithmetic));
        }
      });
    });

    return this;
  }

  sub(other: Matrix<K>): Matrix<K> {
    if (this.size().toString() !== other.size().toString()) {
      throw new Error('Matrices must have the same size');
    }

    this.values.forEach((row, i) => {
      row.forEach((value, j) => {
        const otherValue = other.values[i]![j];

        if (typeof value === 'number' && typeof otherValue === 'number') {
          this.values[i]![j] = (value - otherValue) as K;
        } else if (typeof value === 'bigint' && typeof otherValue === 'bigint') {
          this.values[i]![j] = (value - otherValue) as K;
        } else {
          (this.values[i]![j] as Arithmetic) = (value as Arithmetic).sub((otherValue as Arithmetic));
        }
      });
    });

    return this;
  }

  scl(scalar: K): Matrix<K> {
    this.values.forEach((row, i) => {
      row.forEach((value, j) => {
        if (typeof value === 'number' && typeof scalar === 'number') {
          this.values[i]![j] = (value * scalar) as K;
        } else if (typeof value === 'bigint' && typeof scalar === 'bigint') {
          this.values[i]![j] = (value * scalar) as K;
        } else {
          (this.values[i]![j] as Arithmetic) = (value as Arithmetic).mul(scalar as Arithmetic);
        }
      });
    });

    return this;
  }
}

if (import.meta.vitest) {
  const { it, expect, describe } = import.meta.vitest

  class SomeField implements Arithmetic, Display {
    constructor(public value: number) {}
    add(other: SomeField): SomeField {
      return new SomeField(this.value + other.value);
    }
    sub(other: SomeField): SomeField {
      return new SomeField(this.value - other.value);
    }
    mul(other: SomeField): SomeField {
      return new SomeField(this.value * other.value);
    }
    toString(): string {
      return this.value.toString();
    }
  }

  const matrix = () => new Matrix<number>([[1, 2], [3, 4]])
  const matrixBigInt = () => new Matrix<bigint>([
    [BigInt(1), BigInt(2)],
    [BigInt(3), BigInt(4)]
  ])
  const matrixField = () => new Matrix<SomeField>([
    [new SomeField(1), new SomeField(2)],
    [new SomeField(3), new SomeField(4)]
  ])

  describe('Constructor', () => {
    it('Accepts a 2 dimensional array argument', () => {
      expect(matrix().values).toEqual([[1, 2], [3, 4]])
      expect(matrixBigInt().values).toEqual([
        [BigInt(1), BigInt(2)],
        [BigInt(3), BigInt(4)]
      ])
      expect(matrixField().values).toEqual([
        [new SomeField(1), new SomeField(2)],
        [new SomeField(3), new SomeField(4)]
      ])
    })

    it('validates there is at least 1 row', () => {
      expect(() => new Matrix([])).toThrowError()
      expect(() => new Matrix<bigint>([])).toThrowError()
      expect(() => new Matrix<SomeField>([])).toThrowError()
    })

    it('validates every row is the same length', () => {
      expect(() => new Matrix([[1, 2], [3, 4, 5]])).toThrowError()
    })
  })

  describe('Size', () => {
    it ('returns the size of the matrix', () => {
      expect(matrix().size()).toEqual([2, 2])
      expect(matrixBigInt().size()).toEqual([2, 2])
      expect(matrixField().size()).toEqual([2, 2])
      expect(new Matrix([[1, 2, 3], [4, 5, 6]]).size()).toEqual([2, 3])
      expect(new Matrix([[1, 2], [3, 4], [5, 6]]).size()).toEqual([3, 2])
    })
  })

  describe('Is Square', () => {
    it('returns true if the matrix is square', () => {
      expect(matrix().isSquare()).toEqual(true)
      expect(matrixBigInt().isSquare()).toEqual(true)
      expect(matrixField().isSquare()).toEqual(true)
    })

    it('returns false if the matrix is not square', () => {
      expect(new Matrix([[1, 2], [3, 4], [5, 6]]).isSquare()).toEqual(false)
    })
  })

  describe('toString', () => {
    it('returns a string representation of the matrix', () => {
      expect(matrix().toString()).toEqual('Matrix {\n 1, 2\n 3, 4\n}')
      expect(matrixBigInt().toString()).toEqual('Matrix {\n 1, 2\n 3, 4\n}')
      expect(matrixField().toString()).toEqual('Matrix {\n 1, 2\n 3, 4\n}')
    })
  })

  describe('Add', () => {
    it('adds two matrices together', () => {
      const result = [[2, 4], [6, 8]]
      const resultBigInt = [
        [BigInt(2), BigInt(4)],
        [BigInt(6), BigInt(8)]
      ]
      const resultField = [
        [new SomeField(2), new SomeField(4)],
        [new SomeField(6), new SomeField(8)]
      ]

      expect(matrix().add(matrix()).values).toEqual(result)
      expect(matrixBigInt().add(matrixBigInt()).values).toEqual(resultBigInt)
      expect(matrixField().add(matrixField()).values).toEqual(resultField)
    })

    it('throws an error if the matrices have different sizes', () => {
      expect(() => matrix().add(new Matrix([[1], [2]]))).toThrowError()
      expect(() => matrix().add(new Matrix([[1, 2], [3, 4], [5, 6]]))).toThrowError()
    })
  })

  describe('Sub', () => {
    it('subtracts two matrices', () => {
      const result = [[0, 0], [0, 0]]
      const resultBigInt = [
        [BigInt(0), BigInt(0)],
        [BigInt(0), BigInt(0)]
      ]
      const resultField = [
        [new SomeField(0), new SomeField(0)],
        [new SomeField(0), new SomeField(0)]
      ]

      expect(matrix().sub(matrix()).values).toEqual(result)
      expect(matrixBigInt().sub(matrixBigInt()).values).toEqual(resultBigInt)
      expect(matrixField().sub(matrixField()).values).toEqual(resultField)
    })

    it('throws an error if the matrices have different sizes', () => {
      expect(() => matrix().sub(new Matrix([[1], [2]]))).toThrowError()
    })
  })

  describe('Scl', () => {
    it('multiplies a matrix by a scalar', () => {
      const result = [[2, 4], [6, 8]]
      const resultBigInt = [
        [BigInt(2), BigInt(4)],
        [BigInt(6), BigInt(8)]
      ]
      const resultField = [
        [new SomeField(2), new SomeField(4)],
        [new SomeField(6), new SomeField(8)]
      ]

      expect(matrix().scl(2).values).toEqual(result)
      expect(matrixBigInt().scl(BigInt(2)).values).toEqual(resultBigInt)
      expect(matrixField().scl(new SomeField(2)).values).toEqual(resultField)
    })
  })
}