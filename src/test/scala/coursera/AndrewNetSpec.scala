package coursera

import botkop.nn.coursera.AndrewNet._
import botkop.nn.coursera.{LinearActivationCache, LinearCache}
import numsca.Tensor
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class AndrewNetSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  override def beforeAll(): Unit = {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
  }

  "Linear forward relu" should "activate" in {
    val aPrev = Tensor(-0.41675785, -0.05626683, -2.1361961, 1.64027081,
      -1.79343559, -0.84174737).reshape(3, 2)
    val w = Tensor(0.50288142, -1.24528809, -1.05795222).reshape(1, 3)
    val b = Tensor(-0.90900761)

    val (a, _) = linearActivationForward(aPrev, w, b, reluForward)

    a.shape shouldBe Array(1, 2)
    println(a(0, 0))
    a(0, 0) shouldBe 3.4389 +- 0.0001
    a(0, 1) shouldBe 0.0
  }

  "Linear forward sigmoid" should "activate" in {
    val aPrev = Tensor(-0.41675785, -0.05626683, -2.1361961, 1.64027081,
      -1.79343559, -0.84174737).reshape(3, 2)
    val w = Tensor(0.50288142, -1.24528809, -1.05795222).reshape(1, 3)
    val b = Tensor(-0.90900761)

    val (a, _) = linearActivationForward(aPrev, w, b, sigmoidForward)

    a.shape shouldBe Array(1, 2)
    println(a(0, 0))
    a(0, 0) shouldBe 0.9689 +- 0.0001
    a(0, 1) shouldBe 0.1101 +- 0.0001
  }

  "L model forward" should "activate" in {

    val x = Tensor(1.62434536, -0.61175641, -0.52817175, -1.07296862,
      0.86540763, -2.3015387, 1.74481176, -0.7612069).reshape(4, 2)

    val w1 = Tensor(0.3190391, -0.24937038, 1.46210794, -2.06014071, -0.3224172,
      -0.38405435, 1.13376944, -1.09989127, -0.17242821, -0.87785842,
      0.04221375, 0.58281521).reshape(3, 4)

    val b1 = Tensor(-1.10061918, 1.14472371, 0.90159072).reshape(3, 1)

    val w2 = Tensor(0.50249434, 0.90085595, -0.68372786).reshape(1, 3)

    val b2 = Tensor(-0.12289023)

    val parameters = Map(
      "W1" -> w1,
      "b1" -> b1,
      "W2" -> w2,
      "b2" -> b2
    )

    val (al, caches) = lModelForward(x, parameters)

    al.shape shouldBe Array(1, 2)
    al(0, 0) shouldBe 0.17007265 +- 1e-8
    al(0, 1) shouldBe 0.2524272 +- 1e-7

    caches.length shouldBe 2
  }

  it should "calculate the cross entropy cost" in {
    val y = Tensor(1.0, 1.0, 1.0).reshape(1, 3)
    val yHat = Tensor(0.8, 0.9, 0.4).reshape(1, 3)
    val cost = crossEntropyCost(yHat, y)
    cost shouldBe 0.414931599615 +- 1e-8
  }

  it should "calculate linear backward" in {
    val dz = Tensor(1.62434536, -0.61175641).reshape(1, 2)

    val aPrev = Tensor(-0.52817175, -1.07296862, 0.86540763, -2.3015387,
      1.74481176, -0.7612069).reshape(3, 2)
    val w = Tensor(0.3190391, -0.24937038, 1.46210794).reshape(1, 3)
    val b = Tensor(-2.0601407)

    val cache = new LinearCache(aPrev, w, b)
    val (daPrev, dw, db) = linearBackward(dz, cache)

    val expectedDaPrev = Tensor(0.51822968, -0.19517421, -0.40506361,
      0.15255393, 2.37496825, -0.89445391).reshape(3, 2)
    val expectedDw = Tensor(-0.10076895, 1.40685096, 1.64992505).reshape(1, 3)
    val expectedDb = Tensor(0.50629448)

    approxSameContents(daPrev, expectedDaPrev, 1e-8) shouldBe true
    approxSameContents(dw, expectedDw, 1e-8) shouldBe true
    approxSameContents(db, expectedDb, 1e-8) shouldBe true

  }

  it should "calculate the linear relu activation backward" in {
    val al = Tensor(-0.41675785, -0.05626683).reshape(1, 2)

    val a = Tensor(-2.1361961, 1.64027081, -1.79343559, -0.84174737, 0.50288142,
      -1.24528809).reshape(3, 2)
    val w = Tensor(-1.05795222, -0.90900761, 0.55145404).reshape(1, 3)
    val b = Tensor(2.29220801)
    val linearCache = new LinearCache(a, w, b)
    val activationCache = Tensor(0.04153939, -1.11792545).reshape(1, 2)
    val cache = new LinearActivationCache(linearCache, activationCache)

    val (daPrev, dw, db) = linearActivationBackward(al, cache, reluBackward)

    val daPrevExpected =
      Tensor(0.44090989, 0.0, 0.37883606, 0.0, -0.2298228, 0.0).reshape(3, 2)
    val dwExpected = Tensor(0.44513824, 0.37371418, -0.10478989).reshape(1, 3)
    val dbExpected = Tensor(-0.20837892)

    approxSameContents(daPrev, daPrevExpected, 1e-7) shouldBe true
    approxSameContents(dw, dwExpected, 1e-8) shouldBe true
    approxSameContents(db, dbExpected, 1e-8) shouldBe true
  }

  it should "calculate the linear sigmoid activation backward" in {
    val al = Tensor(-0.41675785, -0.05626683).reshape(1, 2)

    val a = Tensor(-2.1361961, 1.64027081, -1.79343559, -0.84174737, 0.50288142,
      -1.24528809).reshape(3, 2)
    val w = Tensor(-1.05795222, -0.90900761, 0.55145404).reshape(1, 3)
    val b = Tensor(2.29220801)
    val linearCache = new LinearCache(a, w, b)
    val activationCache = Tensor(0.04153939, -1.11792545).reshape(1, 2)
    val cache = new LinearActivationCache(linearCache, activationCache)

    val (daPrev, dw, db) = linearActivationBackward(al, cache, sigmoidBackward)

    val daPrevExpected =
      Tensor(0.11017994, 0.01105339, 0.09466817, 0.00949723, -0.05743092,
        -0.00576154).reshape(3, 2)
    val dwExpected = Tensor(0.10266786, 0.09778551, -0.01968084).reshape(1, 3)
    val dbExpected = Tensor(-0.05729622)

    approxSameContents(daPrev, daPrevExpected, 1e-7) shouldBe true
    approxSameContents(dw, dwExpected, 1e-8) shouldBe true
    approxSameContents(db, dbExpected, 1e-8) shouldBe true
  }

  def approxSameContents(t1: Tensor, t2: Tensor, deviation: Double): Boolean =
    (t1.shape sameElements t2.shape) && {
      val a = t1.array.dup.data.asDouble
      val b = t2.array.dup.data.asDouble
       println(a.toList)
       println(b.toList)
      !a.zip(b).exists {
        case (d1, d2) => math.abs(d2 - d1) > deviation
      }
    }

}
