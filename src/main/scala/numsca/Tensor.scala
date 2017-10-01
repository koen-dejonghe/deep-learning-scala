package numsca

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

import scala.language.postfixOps

class Tensor(val array: INDArray, val isBoolean: Boolean = false)
    extends Serializable {

  def data: Array[Double] = array.dup.data.asDouble

  def shape: Array[Int] = array.shape()
  def reshape(newShape: Array[Int]) = new Tensor(array.reshape(newShape: _*))
  def reshape(newShape: Int*) = new Tensor(array.reshape(newShape: _*))
  def shapeLike(t: Tensor): Tensor = reshape(t.shape)

  def transpose = new Tensor(array.transpose())
  def T: Tensor = transpose

  def round = Tensor(data.map(math.round(_).toDouble)).reshape(this.shape)

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

  def >(d: Double): Tensor = new Tensor(array gt d, true)
  def >=(d: Double): Tensor = new Tensor(array gte d, true)
  def <(d: Double): Tensor = new Tensor(array lt d, true)
  def <=(d: Double): Tensor = new Tensor(array lte d, true)
  def ==(d: Double): Tensor = new Tensor(array eq d, true)
  def !=(d: Double): Tensor = new Tensor(array neq d, true)

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

  def >(other: Tensor): Tensor = new Tensor(array gt bc(other), true)
  def <(other: Tensor): Tensor = new Tensor(array lt bc(other), true)
  def ==(other: Tensor): Tensor = new Tensor(array eq bc(other), true)
  def !=(other: Tensor): Tensor = new Tensor(array neq bc(other), true)

  def maximum(other: Tensor): Tensor =
    new Tensor(Transforms.max(this.array, bc(other)))
  def maximum(d: Double): Tensor = new Tensor(Transforms.max(this.array, d))
  def minimum(other: Tensor): Tensor =
    new Tensor(Transforms.min(this.array, bc(other)))
  def minimum(d: Double): Tensor = new Tensor(Transforms.min(this.array, d))

  private def bc(other: Tensor): INDArray =
    if (sameShape(other))
      other.array
    else
      other.array.broadcast(shape: _*)

  def squeeze(): Double = {
    require(shape sameElements Seq(1, 1))
    array.getDouble(0, 0)
  }

  def apply(index: Int*): Double = array.getDouble(index: _*)
  def apply(index: Array[Int]): Double = apply(index: _*)

  private def indexByBooleanTensor(t: Tensor): Array[Array[Int]] = {
    require(t.isBoolean)
    require(t sameShape this)

    numsca.nditer(t).toArray.flatMap { ii =>
      if (t(ii) == 0) None else Some(ii)
    }
  }

  private def indexByTensor(t: Tensor): Array[Array[Int]] = {
    require(shape.length == t.shape.length)
    require(t.shape.last == 1)
    require(shape.init sameElements t.shape.init)

    numsca.nditer(t).toArray.map { ii =>
      val v = t(ii).toInt
      ii.init :+ v
    }
  }

  private def indexBy(t: Tensor): Array[Array[Int]] =
    if (t.isBoolean) indexByBooleanTensor(t) else indexByTensor(t)

  /*
  slice by tensor
   */
  def apply(t: Tensor): Tensor = {
    val d = indexBy(t).map(apply)
    Tensor(d).reshape(t.shape)
  }

  /*
  this is extremely slow
   */
  def apply(ranges: Seq[Int]*): Tensor = {
    require(ranges.length == shape.length)

    def cross[T](inputs: Seq[Seq[T]]): Seq[Seq[T]] =
      inputs.foldRight(Seq[List[T]](Nil))((el, rest) =>
        el.flatMap(p => rest.map(p :: _)))

    val correctedRanges = ranges.zipWithIndex.map {
      case (r, i) =>
        if (r.isEmpty) 0 until shape(i) else r
    }

    val dta = cross(correctedRanges).map { ii =>
      apply(ii.toArray)
    } toArray

    val newShape = correctedRanges.map(_.length).toArray
    Tensor(dta).reshape(newShape)
  }

  def put(index: Int*)(d: Double): Unit =
    put(index: _*)(d)

  def put(index: Array[Int], d: Double): Unit =
    array.put(NDArrayIndex.indexesFor(index: _*), d)

  def put(t: Tensor, d: Double): Unit =
    indexBy(t).foreach(ix => put(ix, d))

  def put(t: Tensor, f: (Double) => Double): Unit =
    indexBy(t).foreach(ix => put(ix, f(apply(ix))))

  def put(t: Tensor, f: (Array[Int], Double) => Double): Unit =
    indexBy(t).foreach(ix => put(ix, f(ix, apply(ix))))

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
}
