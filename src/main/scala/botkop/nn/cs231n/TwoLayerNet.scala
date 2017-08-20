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

  /* for testing purposes */
  def scores(x: INDArray,
           lw1: INDArray = w1,
           lb1: INDArray = b1,
           lw2: INDArray = w2,
           lb2: INDArray = b2): INDArray = {

    val fa1: (INDArray, Layers.AffineCache) = Layers.affineForward(x, lw1, lb1)
    val rf1: (INDArray, INDArray) = Layers.reluForward(fa1._1)
    val (scores, _): (INDArray, Layers.AffineCache) = Layers.affineForward(rf1._1, lw2, lb2)

    scores
  }

  def loss(x: INDArray,
           y: INDArray,
           lw1: INDArray = w1,
           lb1: INDArray = b1,
           lw2: INDArray = w2,
           lb2: INDArray = b2): (Double, Map[String, INDArray]) = {

    val (af1out, af1cache): (INDArray, Layers.AffineCache) = Layers.affineForward(x, lw1, lb1)
    val (rl1out, rl1cache): (INDArray, INDArray) = Layers.reluForward(af1out)
    val (af2out, af2cache) = Layers.affineForward(rl1out, lw2, lb2)

    val scores = af2out

    val (dataLoss, dout) = Layers.softmaxLoss(scores, y)
    val regLoss = 0.5 * reg * (Nd4j.sum(lw1 mul lw1).getDouble(0) + Nd4j.sum(lw2 mul lw2).getDouble(0))
    val loss = dataLoss + regLoss

    val (dx2, dw2, db2) = Layers.affineBackward(dout, af2cache)
    val da = Layers.reluBackward(dx2, rl1cache)
    val (dx1, dw1, db1) = Layers.affineBackward(da, af1cache)

    val gw2 = dw2 add (lw2 mul reg)
    val gw1 = dw1 add (lw1 mul reg)

    (loss, Map("W2" -> gw2, "b2" -> db2, "W1" -> gw1, "b1" -> db1))
  }


}
