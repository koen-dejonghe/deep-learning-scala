package botkop.nn.jazz

import numsca._


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

}
