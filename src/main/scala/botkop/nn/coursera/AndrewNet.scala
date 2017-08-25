package botkop.nn.coursera

import numsca.Tensor

import scala.language.postfixOps

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
                    b: Tensor): (Tensor, (Tensor, Tensor, Tensor)) = {
    val z = if (b.shape sameElements Array(1, 1)) {
      w.dot(a) + b.data(0)
    } else {
      w.dot(a) + b
    }
    assert(z.shape sameElements Array(w.shape(0), a.shape(1)))
    val cache = (a, w, b)
    (z, cache)
  }

  def linearActivationForward(aPrev: Tensor,
                              w: Tensor,
                              b: Tensor,
                              activation: (Tensor) => (Tensor, Tensor))
    : (Tensor, ((Tensor, Tensor, Tensor), Tensor)) = {
    val (z, linearCache) = linearForward(aPrev, w, b)
    val (a, activationCache) = activation(z)
    val cache = (linearCache, activationCache)
    (a, cache)
  }

  def reluForward(z: Tensor): (Tensor, Tensor) = {
    val a = numsca.maximum(0.0, z)
    (a, z)
  }

  def testLinearForward(): Unit = {
    val m = 3 // num samples
    val ni = 5 // num input features
    val no = 5 // num output features

    val x = numsca.arange(ni * m).reshape(ni, m)
    val w = numsca.ones(no, ni)
    val b = numsca.ones(no, 1)
    val (z, cache) = linearForward(x, w, b)

    assert(z.shape sameElements Array(no, m))
    println(z)
  }

  // testLinearForward()

  def testLinearForwardRelu(): Unit = {
    val aPrev = Tensor(
      Array(-0.41675785, -0.05626683, -2.1361961, 1.64027081, -1.79343559,
        -0.84174737)).reshape(3, 2)
    val w = Tensor(Array(0.50288142, -1.24528809, -1.05795222)).reshape(1, 3)
    val b = Tensor(Array(-0.90900761))

    linearActivationForward(aPrev, w, b, reluForward)
  }

  testLinearForwardRelu()

}
