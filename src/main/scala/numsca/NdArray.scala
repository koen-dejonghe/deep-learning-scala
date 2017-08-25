package numsca

class NdArray (val shape: Array[Int], val data: Array[Double]) {

  require (shape.product == data.length, "shape size and data length do not match")
  require (shape.length >= 2, "shape must contain at least 2 axes")
  require (shape.product > 0, "shape size must be > 0")

  def reshape(newShape: Array[Int]) = new NdArray(newShape, data)


}
