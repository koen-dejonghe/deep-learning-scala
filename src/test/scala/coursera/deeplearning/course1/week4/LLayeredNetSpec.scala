package coursera.deeplearning.course1.week4

import coursera.deeplearning.course1.week4.LLayeredNet._
import numsca.Tensor
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.scalatest.{BeforeAndAfterEach, FlatSpec, Matchers}

import scala.io.Source

class LLayeredNetSpec extends FlatSpec with Matchers with BeforeAndAfterEach {

  override def beforeEach(): Unit = {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    numsca.rand.setSeed(231)
  }

  "An L layered neural net" should "initialize parameters" in {
    val layers = Array(5, 4, 3)
    val parameters = initializeParameters(layers)

    parameters.size shouldBe 4

    parameters("W1").shape shouldBe Array(4, 5)
    parameters("b1").shape shouldBe Array(4, 1)
    parameters("W2").shape shouldBe Array(3, 4)
    parameters("b2").shape shouldBe Array(3, 1)
  }

  it should "activate a linear relu forward layer" in {
    val aPrev = Tensor(-0.41675785, -0.05626683, -2.1361961, 1.64027081,
      -1.79343559, -0.84174737).reshape(3, 2)
    val w = Tensor(0.50288142, -1.24528809, -1.05795222).reshape(1, 3)
    val b = Tensor(-0.90900761)

    val (a, _) = linearActivationForward(aPrev, w, b, reluForward)

    a.shape shouldBe Array(1, 2)
    a(0, 0) shouldBe 3.4389 +- 0.0001
    a(0, 1) shouldBe 0.0
  }

  it should "activate a linear sigmoid forward layer" in {
    val aPrev = Tensor(-0.41675785, -0.05626683, -2.1361961, 1.64027081,
      -1.79343559, -0.84174737).reshape(3, 2)
    val w = Tensor(0.50288142, -1.24528809, -1.05795222).reshape(1, 3)
    val b = Tensor(-0.90900761)

    val (a, _) = linearActivationForward(aPrev, w, b, sigmoidForward)

    a.shape shouldBe Array(1, 2)
    a(0, 0) shouldBe 0.9689 +- 0.0001
    a(0, 1) shouldBe 0.1101 +- 0.0001
  }

  it should "forward propagate an l layered model" in {

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

  it should "backprop a linear layer" in {
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

    approxSameContents(daPrev, expectedDaPrev) shouldBe true
    approxSameContents(dw, expectedDw) shouldBe true
    approxSameContents(db, expectedDb) shouldBe true
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
    approxSameContents(dw, dwExpected) shouldBe true
    approxSameContents(db, dbExpected) shouldBe true
  }

  it should "calculate the linear sigmoid activation backward" in {
    val al = Tensor(-0.41675785, -0.05626683).reshape(1, 2)

    val a = Tensor(-2.1361961, 1.64027081, -1.79343559, -0.84174737, 0.50288142,
      -1.24528809).reshape(3, 2)
    val w = Tensor(-1.05795222, -0.90900761, 0.55145404).reshape(1, 3)
    val b = Tensor(2.29220801)
    val linearCache = new LinearCache(a, w, b)
    // val activationCache = Tensor(0.04153939, -1.11792545).reshape(1, 2)
    val activationCache =
      numsca.sigmoid(Tensor(0.04153939, -1.11792545).reshape(1, 2))
    val cache = new LinearActivationCache(linearCache, activationCache)

    val (daPrev, dw, db) = linearActivationBackward(al, cache, sigmoidBackward)

    val daPrevExpected =
      Tensor(0.11017994, 0.01105339, 0.09466817, 0.00949723, -0.05743092,
        -0.00576154).reshape(3, 2)
    val dwExpected = Tensor(0.10266786, 0.09778551, -0.01968084).reshape(1, 3)
    val dbExpected = Tensor(-0.05729622)

    approxSameContents(daPrev, daPrevExpected) shouldBe true
    approxSameContents(dw, dwExpected) shouldBe true
    approxSameContents(db, dbExpected) shouldBe true
  }

  it should "implement back propagation of an L layered model" in {
    val al = Tensor(1.78862847, 0.43650985).reshape(1, 2)
    val yAssess = Tensor(1, 0).reshape(1, 2)

    val a0 = Tensor(
      0.09649747, -1.8634927, -0.2773882, -0.35475898, -0.08274148, -0.62700068,
      -0.04381817, -0.47721803
    ).reshape(4, 2)
    val w0 = Tensor(
      -1.31386475, 0.88462238, 0.88131804, 1.70957306, 0.05003364, -0.40467741,
      -0.54535995, -1.54647732, 0.98236743, -1.10106763, -1.18504653, -0.2056499
    ).reshape(3, 4)
    val b0 = Tensor(1.48614836, 0.23671627, -1.02378514).reshape(3, 1)
    val linearCache0 = new LinearCache(a0, w0, b0)
    val activationCache0 = Tensor(-0.7129932, 0.62524497, -0.16051336,
      -0.76883635, -0.23003072, 0.74505627).reshape(3, 2)
    val linearActivationCache0 =
      new LinearActivationCache(linearCache0, activationCache0)

    val a1 = Tensor(1.97611078, -1.24412333, -0.62641691, -0.80376609,
      -2.41908317, -0.92379202).reshape(3, 2)
    val w1 = Tensor(-1.02387576, 1.12397796, -0.13191423).reshape(1, 3)
    val b1 = Tensor(-1.62328545)
    val linearCache1 = new LinearCache(a1, w1, b1)
    // val activationCache1 = Tensor(0.64667545, -0.35627076).reshape(1, 2)
    val activationCache1 =
      numsca.sigmoid(Tensor(0.64667545, -0.35627076).reshape(1, 2))
    val linearActivationCache1 =
      new LinearActivationCache(linearCache1, activationCache1)

    val caches = List(linearActivationCache0, linearActivationCache1)

    val (grads, _) = lModelBackward(al, yAssess, caches)

    // println(grads)

    val da1 = Tensor(0, 0.52257901, 0, -0.3269206, 0, -0.32070404, 0,
      -0.74079187).reshape(4, 2)
    val dw1 = Tensor(0.41010002, 0.07807203, 0.13798444, 0.10502167, 0, 0, 0, 0,
      0.05283652, 0.01005865, 0.01777766, 0.0135308).reshape(3, 4)
    val db1 = Tensor(-0.22007063, 0, -0.02835349).reshape(3, 1)

    val da2 = Tensor(0.12913162, -0.44014127, -0.14175655, 0.48317296,
      0.01663708, -0.05670698).reshape(3, 2)
    val dw2 = Tensor(-0.39202432, -0.13325855, -0.04601089).reshape(1, 3)
    val db2 = Tensor(0.15187861)

    approxSameContents(da2, grads("dA2")) shouldBe true
    approxSameContents(dw2, grads("dW2")) shouldBe true
    approxSameContents(db2, grads("db2")) shouldBe true

    approxSameContents(dw1, grads("dW1")) shouldBe true
    approxSameContents(db1, grads("db1")) shouldBe true
    approxSameContents(da1, grads("dA1")) shouldBe true
  }

  it should "update parameters" in {

    val w1 = Tensor(-0.41675785, -0.05626683, -2.1361961, 1.64027081,
      -1.79343559, -0.84174737, 0.50288142, -1.24528809, -1.05795222,
      -0.90900761, 0.55145404, 2.29220801)
      .reshape(4, 3)

    val b1 = Tensor(0.04153939, -1.11792545, 0.53905832).reshape(3, 1)

    val w2 = Tensor(-0.5961597, -0.0191305, 1.17500122).reshape(1, 3)
    val b2 = Tensor(-0.74787095)

    val parameters = Map("W1" -> w1, "b1" -> b1, "W2" -> w2, "b2" -> b2)

    val dw1 = Tensor(1.78862847, 0.43650985, 0.09649747, -1.8634927, -0.2773882,
      -0.35475898, -0.08274148, -0.62700068, -0.04381817, -0.47721803,
      -1.31386475, 0.88462238)
      .reshape(4, 3)
    val db1 = Tensor(0.88131804, 1.70957306, 0.05003364).reshape(3, 1)

    val dw2 = Tensor(-0.40467741, -0.54535995, -1.54647732).reshape(1, 3)
    val db2 = Tensor(0.98236743)

    val grads = Map("dW1" -> dw1, "db1" -> db1, "dW2" -> dw2, "db2" -> db2)

    val updatedParameters = updateParameters(parameters, grads, 0.1)

    println(updatedParameters)

    val w1Expected = Tensor(-0.59562069, -0.09991781, -2.14584584, 1.82662008,
      -1.76569676, -0.80627147, 0.51115557, -1.18258802, -1.0535704,
      -0.86128581, 0.68284052, 2.20374577)
      .reshape(w1.shape)
    val b1Expected = Tensor(-0.04659241, -1.28888275, 0.53405496)
      .reshape(b1.shape)
    val w2Expected =
      Tensor(-0.55569196, 0.0354055, 1.32964895).reshape(w2.shape)
    val b2Expected = Tensor(-0.84610769)

    approxSameContents(updatedParameters("W1"), w1Expected, 1e-7) shouldBe true
    approxSameContents(updatedParameters("b1"), b1Expected, 1e-7) shouldBe true
    approxSameContents(updatedParameters("W2"), w2Expected, 1e-7) shouldBe true
    approxSameContents(updatedParameters("b2"), b2Expected, 1e-7) shouldBe true
  }

  it should "train" in {

    val xTrain = readData("data/coursera/catvsnoncat/train_x.csv", Array(12288, 209))
    val yTrain = readData("data/coursera/catvsnoncat/train_y.csv", Array(1, 209))
    val xTest = readData("data/coursera/catvsnoncat/test_x.csv", Array(12288, 50))
    val yTest = readData("data/coursera/catvsnoncat/test_y.csv", Array(1, 50))

    val layerDims = Array(12288, 20, 7, 5, 1)

    val parameters = lLayerModel(xTrain,
                                 yTrain,
                                 layerDims,
                                 numIterations = 2500,
                                 printCost = true)

    val predTrain = predict(xTrain, yTrain, parameters)
    println(s"training accuracy = $predTrain")

    val predTest = predict(xTest, yTest, parameters)
    println(s"test accuracy = $predTest")

    predTrain should be > 0.9
    predTest should be >= 0.7
  }

  def approxSameContents(t1: Tensor,
                         t2: Tensor,
                         deviation: Double = 1e-8): Boolean =
    (t1 sameShape t2) && {
      val a = t1.array.dup.data.asDouble
      val b = t2.array.dup.data.asDouble
//       println(a.toList)
//       println(b.toList)
      !a.zip(b).exists {
        case (d1, d2) => math.abs(d2 - d1) > deviation
      }
    }

  def readData(fileName: String, shape: Array[Int]): Tensor = {
    val data = Source
      .fromFile(fileName)
      .getLines()
      .map(_.split(",").map(_.toDouble))
      .flatten
      .toArray
    Tensor(data).reshape(shape)
  }

}
