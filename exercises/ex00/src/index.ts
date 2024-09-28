enum Foo {
  a = 'a',
  b = 'b',
  c = 'c',
}

console.log(Object.values(Foo))

if (import.meta.vitest) {
  const { it, expect } = import.meta.vitest

  it('Foo', () => {
    expect(Object.values(Foo)).toEqual(['a', 'b', 'c'])
  })
}