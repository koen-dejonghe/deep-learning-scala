package botkop.nn.cs231n

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object Layers {

  class LayerCache
  case class AffineCache(x: INDArray, w: INDArray, b: INDArray) extends LayerCache

  def affineForward(x: INDArray, w: INDArray, b: INDArray): (INDArray, AffineCache) = {
    val xs = x.reshape(x.shape()(0), w.shape()(0))
    val out = (xs mmul w) addRowVector b
    val cache = AffineCache(x, w, b)
    (out, cache)
  }


  def affineBackward(dout: INDArray, cache: AffineCache): (INDArray, INDArray, INDArray) = {
    import cache._

    val mul = dout mmul w.transpose()
    val dx = mul.reshape(x.shape(): _*)

    val xs = x.reshape(x.shape()(0), w.shape()(0)).transpose()
    val dw = xs mmul dout

    val db = Nd4j.sum(dout, 0)

    (dx, dw, db)
  }

}
