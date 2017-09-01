import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax
import org.nd4j.linalg.api.rng
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.util.Random

package object numsca {

  def array(data: Array[Double], shape: Array[Int]) =
    new Tensor(Nd4j.create(data, shape))

  def create(data: Array[Double], shape: Int*): Tensor =
    array(data, shape.toArray)

  def zeros(shape: Array[Int]) = new Tensor(Nd4j.zeros(shape: _*))

  def zeros(shape: Int*) = new Tensor(Nd4j.zeros(shape: _*))

  def zerosLike(t: Tensor): Tensor = zeros(t.shape)

  def ones(shape: Array[Int]) = new Tensor(Nd4j.ones(shape: _*))

  def ones(shape: Int*) = new Tensor(Nd4j.ones(shape: _*))

  def rand: rng.Random = Nd4j.getRandom

  def randn(shape: Array[Int]) = new Tensor(Nd4j.randn(shape))

  def randn(shape: Int*) = new Tensor(Nd4j.randn(shape.toArray))

  def randint(low: Int, shape: Array[Int]): Tensor = {
    val data = Array.fill(shape.product)(Random.nextInt(low).toDouble)
    array(data, shape)
  }

  def randint(low: Int, shape: Int*): Tensor = randint(low, shape.toArray)

  def linspace(lower: Double, upper: Double, num: Int) =
    new Tensor(Nd4j.linspace(lower, upper, num))

  def abs(t: Tensor) = new Tensor(Transforms.abs(t.array))

  def max(t: Tensor, d: Double) = new Tensor(Transforms.max(t.array, d))
  def maximum(d: Double, t: Tensor) = new Tensor(Transforms.max(t.array, d))
  def max(t: Tensor) = new Tensor(Nd4j.max(t.array))

  def sum(t: Tensor, axis: Int) = new Tensor(Nd4j.sum(t.array, axis))
  def sum(t: Tensor) = new Tensor(Nd4j.sum(t.array))

  def arange(end: Double) = new Tensor(Nd4j.arange(end))
  def arange(start: Double, end: Double) = new Tensor(Nd4j.arange(start, end))

  def sigmoid(t: Tensor): Tensor = new Tensor(Transforms.sigmoid(t.array))
  def relu(t: Tensor): Tensor = new Tensor(Transforms.relu(t.array))
  def tanh(t: Tensor): Tensor = new Tensor(Transforms.tanh(t.array))
  def log(t: Tensor): Tensor = new Tensor(Transforms.log(t.array))
  def power(t: Tensor, power: Double) =
    new Tensor(Transforms.pow(t.array, power))
  def exp(t: Tensor): Tensor = new Tensor(Transforms.exp(t.array))
  def sqrt(t: Tensor): Tensor = new Tensor(Transforms.sqrt(t.array))

  def argmax(t: Tensor, axis: Int = 0): Tensor =
    new Tensor(Nd4j.getExecutioner.exec(new IAMax(t.array), axis))

  def round(t: Tensor): Tensor = new Tensor(Transforms.round(t.array))
  def ceil(t: Tensor): Tensor = new Tensor(Transforms.ceil(t.array))
  def floor(t: Tensor): Tensor = new Tensor(Transforms.floor(t.array))
}
