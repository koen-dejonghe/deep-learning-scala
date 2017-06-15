package botkop.nn

import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.ops.transforms.Transforms.{euclideanDistance, log}
import org.nd4s.Implicits._


trait Cost {
  def function(a: Matrix, y: Matrix): Double
  def delta(a: Matrix, y: Matrix): Matrix
  def name: String
}

object QuadraticCost extends Cost {

  /**
    * Return the cost associated with an output a and the desired output y
    */
  override def function(a: Matrix, y: Matrix): Double = {
    val d: Double = euclideanDistance(a, y)
    0.5 * d * d
  }

  /**
    * Return the error delta from the output layer
    */
  override def delta(a: Matrix, y: Matrix): Matrix = {
    (a - y) * derivative(a)
  }

  def name = "QuadraticCost"

}

object CrossEntropyCost extends Cost {

  /**
    * Return the cost associated with an output a and the desired output y
    */
  override def function(a: Matrix, y: Matrix): Double =
    ((-y * log(a)) - ((-y + 1.0) * log(-a + 1.0))).sum().getDouble(0)

  /**
    * Return the error delta from the output layer
    */
  override def delta(a: Matrix, y: Matrix): Matrix = a - y

  def name = "CrossEntropyCost"
}


