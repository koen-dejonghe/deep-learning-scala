package botkop.nn.cs231n

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}

import scala.language.postfixOps

class TwoLayerNetSpec extends FlatSpec with Matchers {

  // must set data type to double for numerical gradient checking
  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
  Nd4j.getRandom.setSeed(231)

  "A 2 layer net" should "calculate the scores correctly" in {
    val (n, d, h, c) = (3, 5, 50, 7)

    val model = new TwoLayerNet(inputDim = d, hiddenDim = h, numClasses = c)

    val x = Nd4j.linspace(-5.5, 4.5, n * d).reshape(d, n).transpose()
    val w1 = Nd4j.linspace(-0.7, 0.3, d * h).reshape(d, h)
    val b1 = Nd4j.linspace(-0.1, 0.9, h)
    val w2 = Nd4j.linspace(-0.3, 0.4, h * c).reshape(h, c)
    val b2 = Nd4j.linspace(-0.9, 0.1, c)

    val scores = model.scores(x, w1, b1, w2, b2)

    val correctScores = Array(
      11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096,
      12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128, 15.49994135, 16.18839143,
      12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822, 15.66781506, 16.2846319)

    val scoresDiff = scores.data().asDouble().zip(correctScores)
      .map { case (d1, d2) => math.abs(d1 - d2) }.sum

    println(scores)

    scoresDiff should be < 1e-6
  }

  it should "initialize correctly" in {
    val (n, d, h, c) = (3, 5, 50, 7)
    val std = 1e-2
    val model = new TwoLayerNet(inputDim = d, hiddenDim = h, numClasses = c, weightScale = std)

    val w1sd = stdev(model.w1)
    math.abs(w1sd - std) should be < std / 10.0
    val w2sd = stdev(model.w2)
    math.abs(w2sd - std) should be < std / 10.0
    model.b1.data().asDouble().forall(_ == 0.0) should be (true)
    model.b2.data().asDouble().forall(_ == 0.0) should be (true)
  }

  it should "calculate the loss (no regularization)" in {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    val (n, d, h, c) = (3, 5, 50, 7)
    val std = 1e-3
    val model = new TwoLayerNet(inputDim = d, hiddenDim = h, numClasses = c, weightScale = std)

    val x = Nd4j.linspace(-5.5, 4.5, n * d).reshape(d, n).transpose()
    val w1 = Nd4j.linspace(-0.7, 0.3, d * h).reshape(d, h)
    val b1 = Nd4j.linspace(-0.1, 0.9, h)
    val w2 = Nd4j.linspace(-0.3, 0.4, h * c).reshape(h, c)
    val b2 = Nd4j.linspace(-0.9, 0.1, c)
    val y = Nd4j.create(Array(0.0, 5.0, 1.0)).reshape(1, n)

    val (loss, _) = model.loss(x, y, w1, b1, w2, b2)
    val correctLoss = 3.4702243556
    math.abs(loss - correctLoss) should be < 1e-10
  }

  it should "calculate the loss (with regularization)" in {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    val (n, d, h, c) = (3, 5, 50, 7)
    val std = 1e-3
    val model = new TwoLayerNet(inputDim = d, hiddenDim = h, numClasses = c, weightScale = std, reg = 1.0)

    val x = Nd4j.linspace(-5.5, 4.5, n * d).reshape(d, n).transpose()
    val w1 = Nd4j.linspace(-0.7, 0.3, d * h).reshape(d, h)
    val b1 = Nd4j.linspace(-0.1, 0.9, h)
    val w2 = Nd4j.linspace(-0.3, 0.4, h * c).reshape(h, c)
    val b2 = Nd4j.linspace(-0.9, 0.1, c)
    val y = Nd4j.create(Array(0.0, 5.0, 1.0)).reshape(1, n)

    val (loss, _) = model.loss(x, y, w1, b1, w2, b2)
    val correctLoss = 26.5948426952
    math.abs(loss - correctLoss) should be < 1e-10
  }

  def stdev(a: INDArray): Double = {
    val data = a.data().asDouble()
    val mu = data.sum / data.size
    math.sqrt(data.map { d => math.pow(d - mu, 2) }.sum / data.size)
  }

}
