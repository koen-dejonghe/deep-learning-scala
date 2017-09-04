import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax
import org.nd4j.linalg.api.rng
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.language.implicitConversions
import scala.util.Random

package object numsca {

  def rand: rng.Random = Nd4j.getRandom

  def zeros(shape: Int*): Tensor = new Tensor(Nd4j.zeros(shape: _*))
  def zeros(shape: Array[Int]): Tensor = zeros(shape: _*)
  def zerosLike(t: Tensor): Tensor = zeros(t.shape)

  def ones(shape: Int*): Tensor = new Tensor(Nd4j.ones(shape: _*))
  def ones(shape: Array[Int]): Tensor = ones(shape: _*)

  def randn(shape: Int*): Tensor = new Tensor(Nd4j.randn(shape.toArray))
  def randn(shape: Array[Int]): Tensor = randn(shape: _*)

  def randint(low: Int, shape: Array[Int]): Tensor = {
    val data = Array.fill(shape.product)(Random.nextInt(low).toDouble)
    Tensor(data).reshape(shape)
  }
  def randint(low: Int, shape: Int*): Tensor = randint(low, shape.toArray)

  def linspace(lower: Double, upper: Double, num: Int): Tensor =
    new Tensor(Nd4j.linspace(lower, upper, num))

  def abs(t: Tensor): Tensor = new Tensor(Transforms.abs(t.array))

  def maximum(d: Double, t: Tensor): Tensor = new Tensor(Transforms.max(t.array, d))

  def max(t: Tensor): Tensor = new Tensor(Nd4j.max(t.array))

  def sum(t: Tensor, axis: Int): Tensor = new Tensor(Nd4j.sum(t.array, axis))
  def sum(t: Tensor): Tensor = new Tensor(Nd4j.sum(t.array))

  def arange(end: Double): Tensor = new Tensor(Nd4j.arange(end))
  def arange(start: Double, end: Double): Tensor = new Tensor(Nd4j.arange(start, end))

  def sigmoid(t: Tensor): Tensor = new Tensor(Transforms.sigmoid(t.array))
  def relu(t: Tensor): Tensor = new Tensor(Transforms.relu(t.array))
  def tanh(t: Tensor): Tensor = new Tensor(Transforms.tanh(t.array))
  def log(t: Tensor): Tensor = new Tensor(Transforms.log(t.array))
  def power(t: Tensor, pow: Double): Tensor = new Tensor(Transforms.pow(t.array, pow))
  def exp(t: Tensor): Tensor = new Tensor(Transforms.exp(t.array))
  def sqrt(t: Tensor): Tensor = new Tensor(Transforms.sqrt(t.array))
  def square(t: Tensor): Tensor = power(t, 2)

  def argmax(t: Tensor, axis: Int): Tensor =
    new Tensor(Nd4j.getExecutioner.exec(new IAMax(t.array), axis))

  def round(t: Tensor): Tensor = new Tensor(Transforms.round(t.array))
  def ceil(t: Tensor): Tensor = new Tensor(Transforms.ceil(t.array))
  def floor(t: Tensor): Tensor = new Tensor(Transforms.floor(t.array))
}
