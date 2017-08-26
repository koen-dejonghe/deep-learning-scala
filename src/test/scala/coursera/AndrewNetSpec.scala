package coursera

import botkop.nn.coursera.AndrewNet._
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

}
