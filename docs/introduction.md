# Introduction

This project goes through some **Linear Algebra** concepts. The style of project is short form with several exercises.

The project subject code examples are in the [Rust](https://www.rust-lang.org/) language. However, the project allows any programming language as long as it follows a couple of conditions.

- It must support generic types
- It must support functions as first class citizens (example: supports lambda
expressions)
- Optional: support for operator overloading

I used this project to learn `Rust` aswell. I work with full stack `TypeScript` so used the course **Rust for TypeScript Developers** from [Front End Masters](https://frontendmasters.com/courses/rust-ts-devs/) as a starting step. The course was great at making the parallels between the languages.

Using this project as a first application of the newly learned Rust knowledge whilst also doing it in TypeScript just because that comes more naturally and to continue the theme of Rust & TypeScript.

## Assumptions

The project subject mentions this:

> Finally, many of the exercises’ functions store their result within one of their inputs. This is a choice. This type of function can be much faster when using your object as an accumulator, but you might want to have the inputs be immutable instead (ie, a "pure function"), and simply return your value. If your language of choice does not permit such object-oriented constructs, or if you simply prefer the pure function syntax, you are of course allowed to instead have a function that returns a scalar/vector/matrix as output, but keeps its inputs immutable. You may also implement both versions, if you find it worthwhile: the pure function, and the accumulating function. We do ask some amount of coherence though: if you choose to implement only one version, stick to your choice throughout the module.

I have opted to do one version and do the accumulating version, so the object inputs themselves to change. This means that if I have a matrix A and perform an addition on it with another matrix B, matrix A's values will change to the resulting sum.

```ts
const A = new Matrix(...)
const B = new Matrix(...)

A.add(B) // Will change the matrix A values
```
