package cs231n.assignment2

import numsca.Tensor
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.scalatest.{BeforeAndAfterEach, FlatSpec, Matchers}

class LayersSpec extends FlatSpec with Matchers with BeforeAndAfterEach {

  override def beforeEach(): Unit = {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    numsca.rand.setSeed(1)
  }

  "A Layer" should "correctly compute the affine forward pass" in {
    val numInputs = 2
    val inputShape = Array(4, 5, 6)
    val outputDim = 3

    val inputSize = numInputs * inputShape.product
    val weightSize = outputDim * inputShape.product

    val x =
      numsca.linspace(-0.1, 0.5, inputSize).reshape(numInputs +: inputShape)

    val w = numsca
      .linspace(-0.2, 0.3, weightSize)
      .reshape(inputShape.product, outputDim)

    val b = numsca.linspace(-0.3, 0.1, outputDim)

    val out = Layers.affineForward(x, w, b)._1

    val correctOut = Tensor(1.49834967, 1.70660132, 1.91485297, 3.25553199,
      3.5141327, 3.77273342).reshape(numInputs, outputDim)

    val error = relError(out, correctOut)
    error should be < 1e-9
  }

  it should "correctly compute the affine backward pass" in {

    val x = numsca.randn(Array(10, 2, 3))
    val w = numsca.randn(6, 5)
    val b = numsca.randn(1, 5)
    val dout = numsca.randn(10, 5)

    val cache = Layers.affineForward(x, w, b)._2
    val (dx, dw, db) = Layers.affineBackward(dout, cache)

    def fdx(a: Tensor) = Layers.affineForward(a, w, b)._1
    def fdw(a: Tensor) = Layers.affineForward(x, a, b)._1
    def fdb(a: Tensor) = Layers.affineForward(x, w, a)._1

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
    val x = numsca.linspace(-0.5, 0.5, 12).reshape(3, 4)
    val out = Layers.reluForward(x)._1

    val correctOut = Tensor(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04545455,
      0.13636364, 0.22727273, 0.31818182, 0.40909091, 0.5).reshape(x.shape)
    val error = relError(out, correctOut)
    error should be < 6e-8
  }

  it should "correctly compute the relu backward pass" in {
    val x = numsca.randn(10, 10)
    val dout = numsca.randn(x.shape)

    def fdx(a: Tensor) = Layers.reluForward(a)._1

    val dxNum = evalNumericalGradientArray(fdx, x, dout)
    val cache = Layers.reluForward(x)._2
    val dx = Layers.reluBackward(dout, cache)
    val dxError = relError(dx, dxNum)

    dxError should be < 1e-11
  }

  it should "correctly compute the svm loss and gradient" in {
    val numClasses = 10
    val numInputs = 50

    val x = numsca.randn(numInputs, numClasses) * 0.001
    val y = numsca.randint(numClasses, numInputs, 1)

    def fdx(a: Tensor) = Layers.svmLoss(x, y)._1
    val dxNum = evalNumericalGradient(fdx, x)

    val (loss, dx) = Layers.svmLoss(x, y)

    loss should equal (9.0 +- 0.2)

    val dxError = relError(dx, dxNum)
    dxError should be < 1.5e-9
  }

  it should "correctly compute the softmax loss and gradient" in {
    val numClasses = 10
    val numInputs = 50

    val x = numsca.randn(numInputs, numClasses) * 0.001
    val y = numsca.randint(numClasses, numInputs, 1)

    def fdx(a: Tensor) = Layers.softmaxLoss(a, y)._1
    val dxNum: Tensor = evalNumericalGradient(fdx, x)

    val (loss, dx) = Layers.softmaxLoss(x, y)

    loss should equal (2.3 +- 0.2)

    val dxError = relError(dx, dxNum)
    dxError should be < 1e-7
  }

}
