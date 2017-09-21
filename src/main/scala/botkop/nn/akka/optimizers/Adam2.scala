package botkop.nn.akka.optimizers

import numsca.Tensor

import scala.language.postfixOps

case class Adam2(learningRate: Double,
                 beta1: Double = 0.9,
                 beta2: Double = 0.999,
                 epsilon: Double = 1e-8)
    extends Optimizer2[AdamCache] {

  override def localUpdate(
      x: Tensor,
      dx: Tensor,
      maybeCache: Option[AdamCache]): (Tensor, Option[AdamCache]) = {

    val cache = if (maybeCache.isEmpty) AdamCache(x.shape) else maybeCache.get
    val nm = beta1 * cache.m + (1 - beta1) * dx
    val mt = nm / (1 - math.pow(beta1, cache.t))
    val nv = beta2 * cache.v + (1 - beta2) * numsca.square(dx)
    val vt = nv / (1 - math.pow(beta2, cache.t))

    val nextX = x + (-learningRate * mt / numsca.sqrt(vt) + epsilon)
    val nextCache = AdamCache(nm, nv, cache.t + 1)
    (nextX, Some(nextCache))
  }
}

case class AdamCache(m: Tensor, v: Tensor, t: Int) extends OptimizerCache
object AdamCache {
  def apply(shape: Array[Int]): AdamCache =
    AdamCache(numsca.zeros(shape), numsca.zeros(shape), 1)
}
