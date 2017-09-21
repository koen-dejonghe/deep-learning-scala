package botkop.nn.akka.optimizers

import numsca.Tensor

import scala.language.postfixOps

class Adam2(learningRate: Double,
            beta1: Double = 0.9,
            beta2: Double = 0.999,
            epsilon: Double = 1e-8,
) extends Optimizer2[AdamCache] {

  def localUpdate(x: Tensor,
                  dx: Tensor,
                  cache: AdamCache): (Tensor, AdamCache) = {
    val nm = beta1 * cache.m + (1 - beta1) * dx
    val mt = nm / (1 - math.pow(beta1, cache.t))
    val nv = beta2 * cache.v + (1 - beta2) * numsca.square(dx)
    val vt = nv / (1 - math.pow(beta2, cache.t))

    val nextX = x + (-learningRate * mt / numsca.sqrt(vt) + epsilon)
    val nextCache = AdamCache(nm, nv, cache.t + 1)
    (nextX, nextCache)
  }

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor],
                      maybeCaches: Option[List[AdamCache]])
    : (List[Tensor], Option[List[AdamCache]]) = {

    val caches = if (maybeCaches.isEmpty) parameters.map { p =>
      numsca.zerosLike(p)
    } else maybeCaches.get

    val (newXs, newCaches) = parameters
      .zip(gradients)
      .zip(caches)
      .map {
        case ((x: Tensor, dx: Tensor), cache: AdamCache) =>
          localUpdate(x, dx, cache)
      }
      .unzip

    (newXs, Some(newCaches))
  }
}

case class AdamCache(m: Tensor, v: Tensor, t: Int) extends OptimizerCache
