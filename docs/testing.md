# Testing

Each exercise in the project asks for the following:

> You must also turn in a main function in order to test your functions, ready to be compiled (if necessary) and run.

For `TypeScript` use the [vitest](https://vitest.dev/) framework. `Rust` has its own inbuilt automated testing.

## TypeScript

To make the in-source testing work follow the [documentation](https://vitest.dev/guide/in-source).

Make sure your `package.json` has the following line:

```json
"type": "module",
```

And that your `tsconfig.json` has:

```json
"module": "ESNext",
"types": ["vitest/importMeta"],
```

And your `vitest.config.ts` looks like this:

```ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    includeSource: ['src/**/*.ts'],
  },
  define: {
    'import.meta.vitest': 'undefined',
  },
})
```

Finally write an in-source spec like this:

```ts
enum Foo {
  a = 'a',
  b = 'b',
  c = 'c',
}

if (import.meta.vitest) {
  const { it, expect } = import.meta.vitest

  it('Foo', () => {
    expect(Object.values(Foo)).toEqual(['a', 'b', 'c'])
  })
}
```

### Mocks

This is the mock type used as a custom [`Field`](./core-structures).

```typescript
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
```

## Rust

- [Rust Book Chapter 11 on Automated Testing](https://doc.rust-lang.org/book/ch11-00-testing.html)

