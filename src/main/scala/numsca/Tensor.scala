package numsca

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

class Tensor(val array: INDArray) {

  def shape: Array[Int] = array.shape()

  def transpose = new Tensor(array.transpose())
  def T: Tensor = transpose

  def reshape(newShape: Array[Int]) = new Tensor(array.reshape(newShape: _*))
  def reshape(newShape: Int*) = new Tensor(array.reshape(newShape: _*))

  def data: Array[Double] = array.dup.data.asDouble

  def dot(other: Tensor) = new Tensor(array mmul other.array)

  def unary_- : Tensor = new Tensor(array mul -1)
  def +(d: Double) = new Tensor(array add d)
  def -(d: Double) = new Tensor(array sub d)
  def *(d: Double) = new Tensor(array mul d)
  def /(d: Double) = new Tensor(array div d)
  def %(d: Double) = new Tensor(array fmod d)

  def +=(d: Double): Unit = array addi d
  def -=(d: Double): Unit = array subi d
  def *=(d: Double): Unit = array muli d
  def /=(d: Double): Unit = array divi d
  def %=(d: Double): Unit = array fmodi d

  def >(d: Double): Tensor = new Tensor(array gt d)
  def >=(d: Double): Tensor = new Tensor(array gte d)
  def <(d: Double): Tensor = new Tensor(array lt d)
  def <=(d: Double): Tensor = new Tensor(array lte d)
  def ==(d: Double): Tensor = new Tensor(array eq d)
  def !=(d: Double): Tensor = new Tensor(array neq d)

  def +(other: Tensor): Tensor = new Tensor(array add bc(other))
  def -(other: Tensor): Tensor = new Tensor(array sub bc(other))
  def *(other: Tensor): Tensor = new Tensor(array mul bc(other))
  def /(other: Tensor): Tensor = new Tensor(array div bc(other))
  def %(other: Tensor): Tensor = new Tensor(array fmod bc(other))

  def +=(t: Tensor): Unit = array addi bc(t)
  def -=(t: Tensor): Unit = array subi bc(t)
  def *=(t: Tensor): Unit = array muli bc(t)
  def /=(t: Tensor): Unit = array divi bc(t)
  def %=(t: Tensor): Unit = array fmodi bc(t)

  def >(other: Tensor): Tensor = new Tensor(array gt bc(other))
  def <(other: Tensor): Tensor = new Tensor(array lt bc(other))
  def ==(other: Tensor): Tensor = new Tensor(array eq bc(other))
  def !=(other: Tensor): Tensor = new Tensor(array neq bc(other))


  private def bc(other: Tensor): INDArray =
    if (sameShape(other))
      other.array
    else
      other.array.broadcast(shape: _*)

  def squeeze(): Double = {
    require(shape sameElements Seq(1,1))
    array.getDouble(0, 0)
  }


  def apply(index: Int*): Double = array.getDouble(index: _*)
  def apply(index: Array[Int]): Double = array.getDouble(index: _*)

  def put(index: Int*)(d: Double): Unit =
    array.put(NDArrayIndex.indexesFor(index: _*), d)
  def put(index: Array[Int], d: Double): Unit =
    array.put(NDArrayIndex.indexesFor(index: _*), d)

  def sameShape(other: Tensor): Boolean = shape sameElements other.shape
  def sameElements(other: Tensor): Boolean = data sameElements other.data

  override def toString: String = array.toString
}

object Tensor {

  def apply(data: Array[Double]): Tensor = {
    val array = Nd4j.create(data)
    new Tensor(array)
  }

  def apply(data: Double*): Tensor = Tensor(data.toArray)
  def apply(data: Double*)(shape: Int*): Tensor = {
    val t = Tensor.apply(data.toArray)
    t.reshape(shape)
  }

}
