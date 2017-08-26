package botkop.nn.coursera

import numsca.Tensor

import scala.language.postfixOps

class Cache
class LinearForwardCache(a: Tensor, w: Tensor, b: Tensor) extends Cache
class LinearActivationForwardCache(linearCache: LinearForwardCache, a: Tensor)
    extends Cache

object AndrewNet extends App {

  def initializeParametersDeep(layerDims: Array[Int]): Map[String, Tensor] = {
    for {
      l <- 1 until layerDims.length
      w = s"W$l" -> numsca.randn(layerDims(l), layerDims(l - 1)) * 0.01
      b = s"b$l" -> numsca.randn(layerDims(l), 1)
    } yield {
      Seq(w, b)
    }
  }.flatten.toMap

  def linearForward(a: Tensor,
                    w: Tensor,
                    b: Tensor): (Tensor, LinearForwardCache) = {
    val z = w.dot(a) + b
    assert(z.shape sameElements Array(w.shape(0), a.shape(1)))
    val cache = new LinearForwardCache(a, w, b)
    (z, cache)
  }

  def linearActivationForward(aPrev: Tensor,
                              w: Tensor,
                              b: Tensor,
                              activation: (Tensor) => (Tensor, Tensor))
    : (Tensor, LinearActivationForwardCache) = {
    val (z, linearCache) = linearForward(aPrev, w, b)
    val (a, activationCache) = activation(z)
    val cache = new LinearActivationForwardCache(linearCache, activationCache)
    (a, cache)
  }

  def reluForward = (z: Tensor) => {
    val a = numsca.maximum(0.0, z)
    (a, z)
  }

  def sigmoidForward = (z: Tensor) => (numsca.sigmoid(z), z)

  def lModelForward(x: Tensor,
                    parameters: Map[String, Tensor]): (Tensor, List[Cache]) = {
    val numLayers = parameters.size / 2

    (1 to numLayers).foldLeft(x, List.empty[Cache]) {
      case ((aPrev, caches), l) =>
        val w = parameters(s"W$l")
        val b = parameters(s"b$l")
        val activation = if (l == numLayers) sigmoidForward else reluForward
        val (a, cache) = linearActivationForward(aPrev, w, b, activation)
        (a, caches :+ cache)
    }

  }

}
