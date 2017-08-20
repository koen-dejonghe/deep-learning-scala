package numsca

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex

class Tensor(val array: INDArray) {

  def shape: Array[Int] = array.shape()

  def transpose = new Tensor(array.transpose())

  def reshape(newShape: Array[Int]) = new Tensor(array.reshape(newShape: _*))

  def reshape(newShape: Int*) = new Tensor(array.reshape(newShape: _*))

  def data: Array[Double] = array.data().asDouble()

  def dot(other: Tensor) = new Tensor(this.array mmul other.array)

  def +(d: Double) = new Tensor(this.array add d)

  def -(d: Double) = new Tensor(this.array sub d)

  def *(d: Double) = new Tensor(this.array mul d)

  def /(d: Double) = new Tensor(this.array div d)

  def >(d: Double): Tensor = {
    val xs = data.map(x => if (x > d) 1.0 else 0.0)
    numsca.array(xs, this.shape)
  }

  def >=(d: Double): Tensor = {
    val xs = data.map(x => if (x >= d) 1.0 else 0.0)
    numsca.array(xs, this.shape)
  }

  def <(d: Double): Tensor = {
    val xs = data.map(x => if (x < d) 1.0 else 0.0)
    numsca.array(xs, this.shape)
  }

  def <=(d: Double): Tensor = {
    val xs = data.map(x => if (x <= d) 1.0 else 0.0)
    numsca.array(xs, this.shape)
  }

  def +(other: Tensor): Tensor =
    if (this.shape sameElements other.shape) {
      new Tensor(this.array add other.array)
    } else {
      if (other.shape(0) == 1)
        new Tensor(this.array addRowVector other.array)
      else {
        if (other.shape(1) == 1)
          new Tensor(this.array addColumnVector other.array)
        else {
          throw new Exception(s"incompatible shapes ${this.shape.toList} <-> ${other.shape.toList}")
        }
      }
    }

  def -(other: Tensor): Tensor =
    if (this.shape sameElements other.shape) {
      new Tensor(this.array sub other.array)
    } else {
      if (other.shape(0) == 1)
        new Tensor(this.array subRowVector other.array)
      else {
        if (other.shape(1) == 1)
          new Tensor(this.array subColumnVector other.array)
        else {
          throw new Exception(s"incompatible shapes ${this.shape.toList} <-> ${other.shape.toList}")
        }
      }
    }

  def *(other: Tensor): Tensor =
    if (this.shape sameElements other.shape) {
      new Tensor(this.array mul other.array)
    } else {
      if (other.shape(0) == 1)
        new Tensor(this.array mulRowVector other.array)
      else {
        if (other.shape(1) == 1)
          new Tensor(this.array mulColumnVector other.array)
        else {
          throw new Exception(s"incompatible shapes ${this.shape.toList} <-> ${other.shape.toList}")
        }
      }
    }

  def /(other: Tensor): Tensor =
    if (this.shape sameElements other.shape) {
      new Tensor(this.array div other.array)
    } else {
      if (other.shape(0) == 1)
        new Tensor(this.array divRowVector other.array)
      else {
        if (other.shape(1) == 1)
          new Tensor(this.array divColumnVector other.array)
        else {
          throw new Exception(s"incompatible shapes ${this.shape.toList} <-> ${other.shape.toList}")
        }
      }
    }

  def apply(index: Int*): Double = array.getDouble(index: _*)
  def apply(index: Array[Int]): Double = array.getDouble(index: _*)

  def put(index: Int*)(d: Double): Unit = array.put(NDArrayIndex.indexesFor(index: _*), d)
  def put(index: Array[Int], d: Double): Unit = array.put(NDArrayIndex.indexesFor(index: _*), d)

}
