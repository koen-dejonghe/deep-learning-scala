package botkop.nn.coursera

import numsca.Tensor

import scala.language.postfixOps

class Cache
class LinearCache(val a: Tensor, val w: Tensor, val b: Tensor) extends Cache
class LinearActivationCache(val linearCache: LinearCache,
                            val activationCache: Tensor)
    extends Cache

object AndrewNet extends App {

  type ForwardActivationFunction = Tensor => (Tensor, Tensor)
  type BackwardActivationFunction = (Tensor, Tensor) => Tensor

  def initializeParametersDeep(layerDims: Array[Int]): Map[String, Tensor] = {
    for {
      l <- 1 until layerDims.length
      w = s"W$l" -> numsca.randn(layerDims(l), layerDims(l - 1)) * 0.01
      b = s"b$l" -> numsca.randn(layerDims(l), 1)
    } yield {
      Seq(w, b)
    }
  }.flatten.toMap

  def linearForward(a: Tensor, w: Tensor, b: Tensor): (Tensor, LinearCache) = {
    val z = w.dot(a) + b
    assert(z.shape sameElements Array(w.shape(0), a.shape(1)))
    val cache = new LinearCache(a, w, b)
    (z, cache)
  }

  def linearActivationForward(aPrev: Tensor,
                              w: Tensor,
                              b: Tensor,
                              activation: ForwardActivationFunction)
    : (Tensor, LinearActivationCache) = {
    val (z, linearCache) = linearForward(aPrev, w, b)
    val (a, activationCache) = activation(z)
    val cache = new LinearActivationCache(linearCache, activationCache)
    (a, cache)
  }

  def reluForward: ForwardActivationFunction = (z: Tensor) => {
    val a = numsca.maximum(0.0, z)
    (a, z)
  }

  def reluBackward: BackwardActivationFunction =
    (da: Tensor, cache: Tensor) => {
      da * (cache > 0.0)
    }

  def sigmoidForward: ForwardActivationFunction =
    (z: Tensor) => {
      (numsca.sigmoid(z), z)
      // val s = numsca.sigmoid(z)
      // (s, s)
    }

  def sigmoidBackward: BackwardActivationFunction =
    (da: Tensor, cache: Tensor) => {
      val z = cache
      val s = numsca.sigmoid(z)
      val dz = da * s * (-s + 1)
      dz
//      da * cache * (-cache + 1)
    }

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

  def crossEntropyCost(yHat: Tensor, y: Tensor): Double = {
    val m = y.shape(1)
    val logProbs = numsca.log(yHat) * y + (-y + 1.0) * numsca.log(-yHat + 1.0)
    val cost = -numsca.sum(logProbs)(0, 0) / m
    cost
  }

  def linearBackward(dz: Tensor,
                     cache: LinearCache): (Tensor, Tensor, Tensor) = {
    val aPrev = cache.a
    val w = cache.w
    val b = cache.b
    val m = aPrev.shape(1)

    val dw = dz.dot(aPrev.transpose) / m
    val db = numsca.sum(dz, axis = 1) / m
    val daPrev = w.transpose.dot(dz)

    assert(daPrev sameShape aPrev)
    assert(dw sameShape w)
    assert(db sameShape b)

    (daPrev, dw, db)
  }

  def linearActivationBackward(da: Tensor,
                               cache: LinearActivationCache,
                               backwardActivation: BackwardActivationFunction)
    : (Tensor, Tensor, Tensor) = {
    val dz = backwardActivation(da, cache.activationCache)
    val (daPrev, dw, db) = linearBackward(dz, cache.linearCache)

    (daPrev, dw, db)
  }

  def lModelBackward(
      al: Tensor,
      rawY: Tensor,
      caches: List[LinearActivationCache]): Map[String, Tensor] = {
    val numLayers = caches.size
    val y = rawY.reshape(al.shape)

    // derivative of cost with respect to AL
    val dal = -(y / al - (-y + 1) / (-al + 1))

    (1 to numLayers).reverse
      .foldLeft(Map.empty[String, Tensor], dal) {
        case ((grads, da), l) =>
          val currentCache = caches(l - 1)
          val activation =
            if (l == numLayers) sigmoidBackward else reluBackward
          val (daPrev, dw, db) =
            linearActivationBackward(da, currentCache, activation)
          val newMap = grads + (s"dA$l" -> daPrev) + (s"dW$l" -> dw) + (s"db$l" -> db)
          (newMap, daPrev)
      }
      ._1
  }

  def updateParameters(parameters: Map[String, Tensor],
                       grads: Map[String, Tensor],
                       learningRate: Double): Map[String, Tensor] =
    parameters.map {
      case (k, v) =>
        k -> { v - grads(s"d$k") * learningRate }
    }

}
