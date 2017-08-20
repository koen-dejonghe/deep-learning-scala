package botkop.nn.cs231n

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class LayersSpec extends FlatSpec with Matchers {

  // must set data type to double for numerical gradient checking
  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  "A Layer" should "correctly compute the affine forward pass" in {

    val numInputs = 2
    val inputShape = Seq(4, 5, 6)
    val outputDim = 3

    val inputSize = numInputs * inputShape.product
    val weightSize = outputDim * inputShape.product

    val x =
      Nd4j.linspace(-0.1, 0.5, inputSize).reshape(numInputs +: inputShape: _*)
    val w = Nd4j
      .linspace(-0.2, 0.3, weightSize)
      .reshape(inputShape.product, outputDim)
    val b = Nd4j.linspace(-0.3, 0.1, outputDim)

    val out = Layers.affineForward(x, w, b)._1

    val correctData = Array(1.49834967, 1.70660132, 1.91485297, 3.25553199,
      3.5141327, 3.77273342)
    val correctOut = Nd4j.create(correctData).reshape(numInputs, outputDim)

    val error = relError(out, correctOut)
    error should be < 1e-9

  }

  it should "correctly compute the affine backward pass" in {

    Nd4j.getRandom.setSeed(231)
    val x = Nd4j.randn(Array(10, 2, 3))
    val w = Nd4j.randn(6, 5)
    val b = Nd4j.randn(1, 5)
    val dout = Nd4j.randn(10, 5)

    val cache = Layers.affineForward(x, w, b)._2
    val derivatives = Layers.affineBackward(dout, cache)
    val dx = derivatives._1
    val dw = derivatives._2
    val db = derivatives._3

    def fdx(a: INDArray) = Layers.affineForward(a, w, b)._1
    def fdw(a: INDArray) = Layers.affineForward(x, a, b)._1
    def fdb(a: INDArray) = Layers.affineForward(x, w, a)._1

    val dxNum = evalNumericalGradientArray(fdx, x, dout)
    val dwNum = evalNumericalGradientArray(fdw, w, dout)
    val dbNum = evalNumericalGradientArray(fdb, b, dout)

    val dxError = relError(dx, dxNum)
    val dwError = relError(dw, dwNum)
    val dbError = relError(db, dbNum)

    dxError should be < 1e-9
    dwError should be < 1e-9
    dbError should be < 1e-9
  }

  it should "correctly compute the relu forward pass" in {

    val x = Nd4j.linspace(-0.5, 0.5, 12).reshape(3, 4)
    val out = Layers.reluForward(x)._1

    val correctData = Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04545455,
      0.13636364, 0.22727273, 0.31818182, 0.40909091, 0.5)
    val correctOut = Nd4j.create(correctData).reshape(x.shape(): _*)
    val error = relError(out, correctOut)
    error should be < 6e-8
  }

  it should "correctly compute the relu backward pass" in {
    Nd4j.getRandom.setSeed(231)
    val x = Nd4j.randn(10, 10)
    val dout = Nd4j.randn(x.shape)

    def fdx(a: INDArray) = Layers.reluForward(x)._1

    val dxNum = evalNumericalGradientArray(fdx, x, dout)
    val cache = Layers.reluForward(x)._2
    val dx = Layers.reluBackward(dout, cache)
    val dxError = relError(dx, dxNum)

    dxError should be < 1e-11
  }

  it should "correctly compute the svm loss and gradient" in {
    Nd4j.getRandom.setSeed(231)
    val numClasses = 10
    val numInputs = 50

    val x = Nd4j.randn(numInputs, numClasses) mul 0.001
    val y = Nd4j.rand(numClasses, numInputs)

    def fdx(a: INDArray) = Layers.svmLoss(x, y)._1
    val dxNum = evalNumericalGradient(fdx, x)

    val result = Layers.svmLoss(x, y)
    val loss = result._1
    val dx = result._2

    loss should equal (9.0 +- 0.2)

    val dxError = relError(dx, dxNum)
    dxError should be < 1.5e-9
  }

  it should "correctly compute the softmax loss and gradient" in {
    Nd4j.getRandom.setSeed(231)
    val numClasses = 10
    val numInputs = 50

    val x = Nd4j.randn(numInputs, numClasses) mul 0.001

    val yData  = for (_ <- 0 until numInputs) yield Random.nextInt(numClasses).toDouble
    val y = Nd4j.create(yData.toArray).reshape(numInputs, 1)

    def fdx(a: INDArray) = Layers.softmaxLoss(a, y)._1
    val dxNum: INDArray = evalNumericalGradient(fdx, x)

    val (loss, dx) = Layers.softmaxLoss(x, y)

    loss should equal (2.3 +- 0.2)

    val dxError = relError(dx, dxNum)
    dxError should be < 1e-7
  }

}
