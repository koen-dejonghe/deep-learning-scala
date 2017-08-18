package botkop.nn.cs231n

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}

class TwoLayerNetSpec extends FlatSpec with Matchers {

  "A 2 layer net" should "calculate the scores correctly" in {
    Nd4j.getRandom.setSeed(231)
    val (n, d, h, c) = (3, 5, 50, 7)
    // val x: INDArray = Nd4j.randn(Array(n, d))
    val y: INDArray = Nd4j.randn(Array(c, n))

    val std = 1e-3

    val model = new TwoLayerNet(inputDim=d, hiddenDim=h, numClasses=c, weightScale=std)

    val x = Nd4j.linspace(-5.5, 4.5, n*d).reshape(d, n).transpose()
    val scores = model.loss(x)

    println(scores)





  }

}
