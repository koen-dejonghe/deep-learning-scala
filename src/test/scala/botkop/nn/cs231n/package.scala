package botkop.nn

import org.nd4j.linalg.api.iter.NdIndexIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

package object cs231n {

  /**
    * Evaluate a numeric gradient for a function that accepts an array and returns an array.
    */
  def evalNumericalGradientArray(f: (INDArray) => INDArray,
                                 x: INDArray,
                                 df: INDArray,
                                 h: Double = 1e-5): INDArray = {
    val grad = Nd4j.zeros(x.shape(): _*)
    val iter = new NdIndexIterator(x.shape(): _*)
    while (iter.hasNext) {
      val nextIter = iter.next
      val ii = NDArrayIndex.indexesFor(nextIter: _*)

      val oldVal = x.getDouble(nextIter: _*)

      x.put(ii, oldVal + h)
      val pos = f(x)

      x.put(ii, oldVal - h)
      val neg = f(x)

      x.put(ii, oldVal)
      val g = Nd4j.sum((pos sub neg) mul df) div (2.0 * h)
      grad.put(ii, g)
    }
    grad
  }

  def evalNumericalGradient(f: (INDArray) => Double,
                            x: INDArray,
                            h: Double = 0.00001): INDArray = {

    val grad = Nd4j.zeros(x.shape(): _*)
    val iter = new NdIndexIterator(x.shape(): _*)
    while (iter.hasNext) {
      val nextIter = iter.next
      val ii = NDArrayIndex.indexesFor(nextIter: _*)

      val oldVal = x.getDouble(nextIter: _*)

      x.put(ii, oldVal + h)
      val pos = f(x)

      x.put(ii, oldVal - h)
      val neg = f(x)

      x.put(ii, oldVal)

      val g = (pos - neg) / (2.0 * h)
      grad.put(ii, g)
    }

    grad
  }

  /**
    * returns relative error
    */
  def relError(x: INDArray, y: INDArray): Double = {
    import org.nd4j.linalg.ops.transforms.Transforms._
    val n = abs(x sub y)

    val d = max(abs(x.dup()) add abs(y.dup()), 1e-8)
    Nd4j.max(n div d).getDouble(0)
  }

}
