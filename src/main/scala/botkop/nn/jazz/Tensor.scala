package botkop.nn.jazz

import scala.annotation.tailrec
import scala.language.postfixOps
import scala.util.Random

class Tensor(val data: Array[Double], val shape: Array[Int]) {
  require(shape.product == data.length)
  def this(data: Array[Double], shape: Int*) = this(data, shape.toArray)
  val size: Int = data.length

  def reshape(newShape: Array[Int]) = new Tensor(data, newShape)
  def reshape(newShape: Int*) = new Tensor(data, newShape.toArray)
  def t = new Tensor(data, shape.tail :+ shape.head)

  def mergeData[T](other: Tensor, f: (Double, Double) => T): Seq[T] = {
    require(this.shape sameElements other.shape)
    data
      .zip(other.data)
      .map { case (a, b) => f(a, b) }
  }

  def merge(other: Tensor, f: (Double, Double) => Double): Tensor = {
    val d = mergeData(other, f)
    new Tensor(d.toArray, this.shape)
  }

  def +(other: Tensor): Tensor = merge(other, _ + _)
  def -(other: Tensor): Tensor = merge(other, _ - _)
  def *(other: Tensor): Tensor = merge(other, _ * _)
  def /(other: Tensor): Tensor = merge(other, _ / _)

  def dot(other: Tensor): Tensor = {
    require(this.shape(1) == other.shape(0))
    require(this.shape.length == 2 && other.shape.length == 2)

    val v1 = this.data.grouped(this.shape(1))
    val v2 = other.data.grouped(other.shape(0))

    val d = v1.zip(v2).map {
      case (a1, a2) =>
        a1.zip(a2).map { case (d1, d2) =>
          println(s"$d1 * $d2")
          d1 * d2
        } sum
    }
    new Tensor(d.toArray, this.shape(0), other.shape(1))
  }

  def +(d: Double): Tensor = new Tensor(data.map(_ + d), shape)
  def -(d: Double): Tensor = new Tensor(data.map(_ - d), shape)
  def *(d: Double): Tensor = new Tensor(data.map(_ * d), shape)
  def /(d: Double): Tensor = new Tensor(data.map(_ / d), shape)

  def sum: Double = data.sum

  @tailrec
  final def getElement(index: List[Int],
                       shapeIndex: Int = 0,
                       d: Array[Double] = data): Double = index match {
    case Nil =>
      throw new Exception("index cannot be empty")
    case i :: Nil =>
      data(i)
    case i :: is =>
      val gs = d.grouped(shape(shapeIndex)).toArray
      getElement(is, shapeIndex + 1, gs(i))
  }

  def apply(index: Int*): Double = {
    require(index.length == shape.length)
    for ((s, i) <- shape.zip(index)) {
      require(s > i)
      require(i >= 0)
    }
    getElement(index.toList)
  }

}

object Tensor {

  def zeros(shape: Int*): Tensor = {
    new Tensor(Array.fill(shape.product)(0.0), shape.toArray)
  }
  def ones(shape: Int*): Tensor =
    new Tensor(Array.fill(shape.product)(1.0), shape.toArray)
  def randn(shape: Int*): Tensor = {
    val r = new Random()
    new Tensor(Array.fill(shape.product) { r.nextGaussian() }, shape.toArray)
  }

}
