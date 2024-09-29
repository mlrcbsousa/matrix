export interface Display {
  toString(): string;
}

export interface Arithmetic {
  add(other: Arithmetic): Arithmetic;
  sub(other: Arithmetic): Arithmetic;
  mul(other: Arithmetic): Arithmetic;
}

export type Field = number | bigint | Arithmetic;

export interface Structure<K, T> {
  add(other: T): T;
  sub(other: T): T;
  scl(scalar: K): T;
}
