package botkop.nn.jazz

import numsca._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


object Layers {

  case class AffineCache(x: Tensor, w: Tensor, b: Tensor)
  case class ReluCache(x: Tensor)

  def affineForward(x: Tensor, w: Tensor, b: Tensor): (Tensor, AffineCache) = {
    val xs = x.reshape(x.shape(0), w.shape(0))
    val out = (xs dot w) + b
    val cache = AffineCache(x, w, b)
    (out, cache)
  }

  def affineBackward(dout: Tensor,
                     cache: AffineCache): (Tensor, Tensor, Tensor) = {
    import cache._

    val mul = dout dot w.transpose
    val dx = mul.reshape(x.shape)

    val xs = x.reshape(x.shape(0), w.shape(0)).transpose
    val dw = xs dot dout

    val db = sum(dout, axis=0)

    (dx, dw, db)
  }

  def reluForward(x: Tensor): (Tensor, ReluCache) = {
    val out = max(x, 0.0)
    val cache = ReluCache(x)
    (out, cache)
  }

  def reluBackward(dout: Tensor, cache: ReluCache): Tensor = {
    import cache._
    val dx = dout * (x > 0.0)
    dx
  }

  def svmLoss(x: Tensor, y: Tensor): (Double, Tensor) = {
    val n = x.shape(0)
    val correctClassScores = x(arange(n))(y)
    val margins = maximum(0.0, x - correctClassScores + 1.0)
    margins(arange(n))(y).put(0.0)
    val loss = sum(margins)(0) / n
    val numPos = sum(margins > 0, axis=1)(0)
    val dx = zerosLike(x)
    dx(margins > 0).put(1.0)
    dx(arange(n))(y) -= numPos
    dx /= n
    (loss, dx)
  }


}
