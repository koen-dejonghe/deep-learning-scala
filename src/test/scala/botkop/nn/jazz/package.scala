package botkop.nn

import numsca._
import org.nd4j.linalg.api.iter.NdIndexIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

package object jazz {
  /**
    * returns relative error
    */
  def relError(x: Tensor, y: Tensor): Double = {
    val n = abs(x - y)
    val d = max(abs(x) + abs(y), 1e-8)
    max(n / d)(0)
  }

  /**
    * Evaluate a numeric gradient for a function that accepts an array and returns an array.
    */
  def evalNumericalGradientArray(f: (Tensor) => Tensor,
                                 x: Tensor,
                                 df: Tensor,
                                 h: Double = 1e-5): Tensor = {
    val grad = numsca.zerosLike(x)
    val iter = new NdIndexIterator(x.shape: _*)
    while (iter.hasNext) {
      val ix: Array[Int] = iter.next

      val oldVal = x(ix)
      x.put(ix, oldVal + h)
      val pos = f(x)
      x.put(ix, oldVal - h)
      val neg = f(x)
      x.put(ix, oldVal)

      val g: Double = (sum((pos - neg) * df) / (2.0 * h))(0)
      grad.put(ix, g)
    }
    grad
  }

  def evalNumericalGradient(f: (Tensor) => Double,
                            x: Tensor,
                            h: Double = 0.00001): Tensor = {

    val grad = zeros(x.shape)
    val iter = new NdIndexIterator(x.shape: _*)
    while (iter.hasNext) {
      val nextIter = iter.next

      val oldVal = x(nextIter)
      x.put(nextIter, oldVal + h)
      val pos = f(x)
      x.put(nextIter, oldVal - h)
      val neg = f(x)
      x.put(nextIter, oldVal)
      val g = (pos - neg) / (2.0 * h)
      grad.put(nextIter, g)
    }

    grad
  }


}
