import { Structure, Field, Arithmetic } from "./types.ts";

export default class Vector<K extends Field> implements Structure<K, Vector<K>> {
  // TODO: A function to reshape a vector into a matrix, and vice-versa.

  constructor(public values: K[]) {}

  size(): number {
    return this.values.length;
  }

  toString(): string {
    return `Vector { ${this.values.join(', ')} }`;
  }

  add(other: Vector<K>): Vector<K> {
    if (this.size() !== other.size()) {
      throw new Error('Vectors must have the same size');
    }

    this.values.forEach((value, i) => {
      const otherValue = other.values[i];

      if (typeof value === 'number' && typeof otherValue === 'number') {
        this.values[i] = (value + otherValue) as K;
      } else if (typeof value === 'bigint' && typeof otherValue === 'bigint') {
        this.values[i] = (value + otherValue) as K;
      } else {
        (this.values[i] as Arithmetic) = (value as Arithmetic).add((otherValue as Arithmetic));
      }
    });

    return this;
  }

  sub(other: Vector<K>): Vector<K> {
    if (this.size() !== other.size()) {
      throw new Error('Vectors must have the same size');
    }

    this.values.forEach((value, i) => {
      const otherValue = other.values[i];

      if (typeof value === 'number' && typeof otherValue === 'number') {
        this.values[i] = (value - otherValue) as K;
      } else if (typeof value === 'bigint' && typeof otherValue === 'bigint') {
        this.values[i] = (value - otherValue) as K;
      } else {
        (this.values[i] as Arithmetic) = (value as Arithmetic).sub((otherValue as Arithmetic));
      }
    });

    return this;
  }

  scl(scalar: K): Vector<K> {
    this.values.forEach((value, i) => {
      if (typeof value === 'number' && typeof scalar === 'number') {
        this.values[i] = (value * scalar) as K;
      } else if (typeof value === 'bigint' && typeof scalar === 'bigint') {
        this.values[i] = (value * scalar) as K;
      } else {
        (this.values[i] as Arithmetic) = (value as Arithmetic).mul(scalar as Arithmetic);
      }
    });

    return this;
  }
}

if (import.meta.vitest) {
  const { it, expect, describe } = import.meta.vitest

  const vector = () => new Vector<number>([1, 2])

  describe('Constructor', () => {
    it('Accepts an array argument', () => {
      expect(vector().values).toEqual([1, 2])
    })
  })

  describe('Size', () => {
    it ('returns the size of the vector', () => {
      expect(vector().size()).toEqual(2)
    })
  })

  describe('toString', () => {
    it('returns a string representation of the vector', () => {
      expect(vector().toString()).toEqual('Vector { 1, 2 }')
    })
  })

  describe('Add', () => {
    it('adds two vectors together', () => {
      const result = [4, 6]
      const other = new Vector([3, 4])

      expect(vector().add(other).values).toEqual(result)
    })

    it('throws an error if the vectors have different sizes', () => {
      expect(() => vector().add(new Vector([1, 2, 3, 4]))).toThrowError()
    })
  })

  describe('Sub', () => {
    it('subtracts two vectors', () => {
      const result = [0, 0]

      expect(vector().sub(vector()).values).toEqual(result)
    })

    it('throws an error if the vectors have different sizes', () => {
      expect(() => vector().sub(new Vector([3, 4, 5]))).toThrowError()
    })
  })

  describe('Scl', () => {
    it('multiplies a vector by a scalar', () => {
      const result = [2, 4]

      expect(vector().scl(2).values).toEqual(result)
    })
  })
}