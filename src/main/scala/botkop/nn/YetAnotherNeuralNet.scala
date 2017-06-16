package botkop.nn

import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.language.postfixOps
import scala.util.Random

case class Layer(biases: Matrix, weights: Matrix) {
  def activate(x: Matrix): Matrix = sigmoid((weights dot x) + biases)

  def +=(other: Layer): Unit = {
    biases += other.biases
    weights += other.weights
  }

}

case class Network(layers: List[Layer]) {

  def size: Int = layers.size

  def weights: List[Matrix] = layers.map(_.weights)
  def biases: List[Matrix] = layers.map(_.biases)

  def feedForward(x: Matrix): List[Matrix] = layers.foldLeft(List(x)) {
    case (activations, layer) =>
      activations :+ layer.activate(activations.last)
  }

  def backProp(x: Matrix, y: Matrix): Network = {
    val activations = feedForward(x)

    val delta = activations.last - y

    val inb = delta
    val inw = delta dot activations(activations.size - 2).transpose()

    val deltaLayers = (size - 1 until 0 by -1)
      .foldLeft(List(Layer(inb, inw))) {
        case (nablaLayerAcc, layerIndex) =>
          val sp = derivative(activations(layerIndex))

          // last added nb to nbl is the previous delta
          val delta = (weights(layerIndex)
            .transpose() dot nablaLayerAcc.head.biases) * sp

          val nb = delta
          val nw = delta dot activations(layerIndex - 1).transpose()

          Layer(nb, nw) :: nablaLayerAcc
      }
    Network(deltaLayers)
  }

  def accuracy(data: List[(Matrix, Matrix)]): Int = data.foldLeft(0) {
    case (r, (x, y)) =>
      val a = feedForward(x).last
      val guess = argMax(a).getInt(0)
      val truth = argMax(y).getInt(0)
      if (guess == truth) r + 1 else r
  }

  def zeroize: Network = {
    val zl = layers.map { layer =>
      val zb = zeros(layer.biases.shape(): _*)
      val zw = zeros(layer.weights.shape(): _*)
      Layer(zb, zw)
    }
    Network(zl)
  }

  def +=(other: Network): Unit =
    layers.zip(other.layers).foreach { case (l1, l2) => l1 += l2 }

}

object Network {

  def squash(biases: List[Matrix], weights: List[Matrix]): Network = {
    val layers = biases.zip(weights).map { case (b, w) => Layer(b, w) }
    Network(layers)
  }

  def init(topology: List[Int]): Network = {
    val (biases, weights) = initializeBiasesAndWeights(topology)
    squash(biases, weights)
  }

}

class YetAnotherNeuralNet(network: Network) {

  private val zeroNet = network.zeroize

  def updateMiniBatch(miniBatch: List[(Matrix, Matrix)],
                      lm: Double,
                      lln: Double): Unit = {

    val nablaNet = zeroNet

    miniBatch.foreach {
      case (x, y) =>
        nablaNet += network.backProp(x, y)
    }

    network.biases.zip(nablaNet.biases).foreach {
      case (b, nb) =>
        b -= nb * lm
    }

    network.weights.zip(nablaNet.weights).foreach {
      case (w, nw) =>
        w *= lln
        w -= nw * lm
    }

  }

  def sgd(trainingData: List[(Matrix, Matrix)],
          epochs: Int,
          miniBatchSize: Int,
          learningRate: Double,
          lambda: Double,
          evaluationData: List[(Matrix, Matrix)] = List.empty,
          monitorEvaluationCost: Boolean = false,
          monitorEvaluationAccuracy: Boolean = false,
          monitorTrainingCost: Boolean = false,
          monitorTrainingAccuracy: Boolean = false): Monitor = {

    val monitor = Monitor()

    val lm = learningRate / miniBatchSize
    val lln = 1.0 - learningRate * (lambda / trainingData.size)

    (1 to epochs).foreach { epoch =>
      val t0 = System.currentTimeMillis()
      val shuffled = Random.shuffle(trainingData)
      shuffled.sliding(miniBatchSize, miniBatchSize).foreach { miniBatch =>
        updateMiniBatch(miniBatch, lm, lln)
      }
      val t1 = System.currentTimeMillis()
      println(s"Epoch $epoch completed in ${t1 - t0} ms.")

      if (monitorEvaluationAccuracy) {
        val a = network.accuracy(evaluationData)
        println(s"Accuracy on evaluation data: $a / ${evaluationData.size}")
        monitor.evaluationAccuracy += a
      }
    }
    monitor
  }
}

object YetAnotherNeuralNet {

  def main(args: Array[String]) {
    val topology = List(784, 30, 30, 10)
    val epochs = 30
    val batchSize = 10
    val learningRate = 0.5
    val lambda = 0.5

    val nw = Network.init(topology)
    val nn = new YetAnotherNeuralNet(nw)

    val (trainingData, validationData, testData) = mnistData()

    val monitor = nn.sgd(
      trainingData,
      epochs,
      batchSize,
      learningRate,
      lambda,
      validationData,
      monitorEvaluationAccuracy = true
    )

    println(monitor)

  }

}
