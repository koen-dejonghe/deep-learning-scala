package botkop.nn.coursera

import numsca.Tensor

import scala.io.Source
import scala.language.postfixOps

object AndrewNet extends App {

  class Cache
  class LinearCache(val a: Tensor, val w: Tensor, val b: Tensor) extends Cache
  class LinearActivationCache(val linearCache: LinearCache,
                              val activationCache: Tensor)
      extends Cache

  type ForwardActivationFunction = Tensor => (Tensor, Tensor)
  type BackwardActivationFunction = (Tensor, Tensor) => Tensor

  def initializeParameters(layerDims: Array[Int]): Map[String, Tensor] =
    (1 until layerDims.length).foldLeft(Map.empty[String, Tensor]) {
      case (parameters, l) =>
        val w = numsca.randn(layerDims(l), layerDims(l - 1)) / math.sqrt(layerDims(l-1))
        val b = numsca.zeros(layerDims(l), 1)
        parameters ++ Seq(s"W$l" -> w, s"b$l" -> b)
    }

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

  def lModelForward(x: Tensor, parameters: Map[String, Tensor])
    : (Tensor, List[LinearActivationCache]) = {
    val numLayers = parameters.size / 2

    (1 to numLayers).foldLeft(x, List.empty[LinearActivationCache]) {
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

    val logProbs = numsca.log(yHat) * y + (-y + 1) * numsca.log(-yHat + 1)
    val cost = -numsca.sum(logProbs)(0, 0) / m
    cost

    // cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    // val cost = (-y.dot(numsca.log(yHat).transpose) - (-y + 1).dot(numsca.log(-yHat + 1).transpose)) / m
    // cost(0,0)

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
      caches: List[LinearActivationCache]): (Map[String, Tensor], Tensor) = {
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
          val newGrads = grads + (s"dA$l" -> daPrev) + (s"dW$l" -> dw) + (s"db$l" -> db)
          (newGrads, daPrev)
      }
  }

  def updateParameters(parameters: Map[String, Tensor],
                       grads: Map[String, Tensor],
                       learningRate: Double): Map[String, Tensor] =
    parameters.map {
      case (k, v) =>
        k -> { v - (grads(s"d$k") * learningRate) }
    }

  def lLayerModel(x: Tensor,
                  y: Tensor,
                  layerDims: Array[Int],
                  learningRate: Double = 0.0075,
                  numIterations: Int = 3000,
                  printCost: Boolean = false): Map[String, Tensor] = {

    val initialParameters = initializeParameters(layerDims)

    (1 to numIterations).foldLeft(initialParameters) {
      case (parameters, i) =>
        val (al, caches) = lModelForward(x, parameters)
        val cost = crossEntropyCost(al, y)
        if (printCost && i % 100 == 0) println(s"iteration $i: cost = $cost")
        val (grads, _) = lModelBackward(al, y, caches)
        updateParameters(parameters, grads, learningRate)
    }
  }

  def readData(fileName: String, shape: Array[Int]): Tensor = {
    val data = Source
      .fromFile(fileName)
      .getLines()
      .map(_.split(",").map(_.toDouble))
      .flatten
      .toArray
    numsca.array(data, shape)
  }

  def predict(x: Tensor, y: Tensor, parameters: Map[String, Tensor]): Double = {
    val m = x.shape(1)
    val n = parameters.size / 2

    val (probas, _) = lModelForward(x, parameters)

    val p = probas > 0.5
    val accuracy = numsca.sum(p == y) / m
    accuracy(0, 0)
  }

}
