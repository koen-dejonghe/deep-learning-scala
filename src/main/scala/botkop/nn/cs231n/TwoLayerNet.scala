package botkop.nn.cs231n

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class TwoLayerNet(inputDim: Int = 3 * 32 * 32,
                  hiddenDim: Int = 100,
                  numClasses: Int = 10,
                  weightScale: Double = 1e-3,
                  reg: Double = 0.0) {

  val w1: INDArray = Nd4j.randn(Array(inputDim, hiddenDim)) mul weightScale
  val b1: INDArray = Nd4j.zeros(hiddenDim)
  val w2: INDArray = Nd4j.randn(Array(hiddenDim, numClasses)) mul weightScale
  val b2: INDArray = Nd4j.zeros(numClasses)

  def loss(x: INDArray, y: Option[INDArray] = None): INDArray = {

    val fa1: (INDArray, Layers.AffineCache) = Layers.affineForward(x, w1, b1)
    val rf1: (INDArray, INDArray) = Layers.reluForward(fa1._1)
    val fa2: (INDArray, Layers.AffineCache) = Layers.affineForward(rf1._1, w2, b2)
    val rf2: (INDArray, INDArray) = Layers.reluForward(fa2._1)

    val scores = rf2._2

    scores

  }


}
