import Matrix from "./structures/Matrix.class.ts"
import Vector from "./structures/Vector.class.ts"

function main() {
  const matrix = new Matrix<number>([[1, 2], [3, 4]])
  const matrixOther = new Matrix<number>([[5, 6], [7, 8]])

  console.log(`Print: ${matrix}`)
  console.log(`Values: ${matrix.values}`)
  console.log(`Size: ${matrix.size()}`)
  console.log(`Add: ${matrix.add(matrixOther).values}`)
  console.log(`Subtract: ${matrix.sub(matrixOther).values}`)
  console.log(`Scale: ${matrix.scl(2).values}`)

  console.log('---')

  const vector = new Vector<number>([1, 2])
  const vectorOther = new Vector<number>([3, 4])

  console.log(`Print: ${vector}`)
  console.log(`Values: ${vector.values}`)
  console.log(`Size: ${vector.size()}`)
  console.log(`Add: ${vector.add(vectorOther).values}`)
  console.log(`Subtract: ${vector.sub(vectorOther).values}`)
  console.log(`Scale: ${vector.scl(2).values}`)
}

main()