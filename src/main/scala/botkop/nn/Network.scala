package botkop.nn

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.io.Source
import scala.language.postfixOps
import scala.util.Random

trait Cost {
  def function(a: Matrix, y: Matrix): Double
  def delta(a: Matrix, y: Matrix): Matrix
}

object QuadraticCost extends Cost {

  import Network._

  /**
    * Return the cost associated with an output a and the desired output y
    */
  override def function(a: Matrix, y: Matrix): Double = {
    val d: Double = euclideanDistance(a, y)
    0.5 * d * d
  }

  /**
    * Return the error delta from the output layer
    */
  override def delta(a: Matrix, y: Matrix): Matrix = {
    (a - y) * derivative(a)
  }

}

object CrossEntropyCost extends Cost {

  /**
    * Return the cost associated with an output a and the desired output y
    */
  override def function(a: Matrix, y: Matrix): Double =
    ((-y * log(a)) - ((-y + 1.0) * log(-a + 1.0))).sum().getDouble(0)

  /**
    * Return the error delta from the output layer
    */
  override def delta(a: Matrix, y: Matrix): Matrix = a - y
}

class Network(topology: List[Int], cost: Cost) {

  import Network._

  val (biases, weights) = initialize(topology)

  /**
    * Initialize each weight using a Gaussian distribution with mean 0
    * and standard deviation 1 over the square root of the number of
    * weights connecting to the same neuron.  Initialize the biases
    * using a Gaussian distribution with mean 0 and standard
    * deviation 1.
    * Note that the first layer is assumed to be an input layer, and
    * by convention we won't set any biases for those neurons, since
    * biases are only ever used in computing the outputs from later
    * layers.
    * @param topology layer definition of the network
    * @return tuple of (biases, weights)
    */
  def initialize(topology: List[Int]): (List[Matrix], List[Matrix]) = {

    val biases: List[Matrix] =
      topology.tail.map(size => randn(size, 1))

    val weights: List[Matrix] = topology
      .sliding(2)
      .map(t => (t.head, t(1)))
      .map { case (x, y) => randn(y, x) / Math.sqrt(x) }
      .toList

    (biases, weights)
  }

  /**
    * Return the output of the network if x is input
    * @param x input to the network
    * @return all activated layers
    */
  def feedForward(x: Matrix): List[Matrix] = {
    biases.zip(weights).foldLeft(List(x)) {
      case (as, (b, w)) =>
        val z = (w dot as.last) + b
        val a = sigmoid(z)
        as :+ a
    }
  }

  def backProp(x: Matrix, y: Matrix): (List[Matrix], List[Matrix]) = {

    val activations = feedForward(x)

    val delta = cost.delta(activations.last, y)

    val inb = delta
    val inw = delta dot activations(activations.size - 2).transpose()

    val (nablaBiases, nablaWeights) = (topology.size - 2 until 0 by -1)
      .foldLeft((List(inb), List(inw))) {
        case ((nbl, nwl), l) =>
          val sp = derivative(activations(l))

          // last added nb to nbl is the previous delta
          val delta = (weights(l).transpose() dot nbl.head) * sp

          val nb = delta
          val nw = delta dot activations(l - 1).transpose()

          (nb :: nbl, nw :: nwl)
      }

    (nablaBiases, nablaWeights)
  }

  def updateMiniBatch(miniBatch: List[(Matrix, Matrix)],
                      learningRate: Double,
                      lambda: Double,
                      n: Int): Unit = {

    val nablaBiases = biases.map(b => zeros(b.shape(): _*))
    val nablaWeights = weights.map(w => zeros(w.shape(): _*))

    miniBatch.foreach {
      case (x, y) =>
        val (deltaNablaB, deltaNablaW) = backProp(x, y)

        nablaBiases.zip(deltaNablaB).foreach {
          case (nb, dnb) =>
            nb += dnb
        }

        nablaWeights.zip(deltaNablaW).foreach {
          case (nw, dnw) =>
            nw += dnw
        }
    }

    val m = miniBatch.size
    val lm = learningRate / m
    val lln = 1.0 - learningRate * (lambda / n)

    biases.zip(nablaBiases).foreach {
      case (b, nb) =>
        b -= nb * lm
    }

    weights.zip(nablaWeights).foreach {
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
          evaluationData: List[(Matrix, Int)] = List.empty): Unit = {

    (1 to epochs).foreach { j =>
      val t0 = System.currentTimeMillis()
      println(s"Epoch $j starting")

      val shuffled = Random.shuffle(trainingData)
      shuffled.sliding(miniBatchSize, miniBatchSize).foreach { miniBatch =>
        updateMiniBatch(miniBatch, learningRate, lambda, trainingData.size)
      }
      val t1 = System.currentTimeMillis()

      println(s"Epoch $j complete (${t1 - t0})")
    }
  }

}

object Network {

  def derivative(z: Matrix): Matrix = z * (-z + 1.0)

  def oneHotEncoded(x: Int, base: Int = 10): Matrix = {
    val v = zeros(1, base)
    v.putScalar(x, 1.0)
  }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String): List[(Matrix, Int)] = {
    Source.fromInputStream(gzis(fname)).getLines() map { line =>
      val tokens = line.split(",")
      val (y, x) = (tokens.head.toInt, tokens.tail.map(_.toDouble / 255.0))
      (create(x).transpose(), y)
    } toList
  }

  def main(args: Array[String]) {
    val topology = List(784, 30, 30, 10)
    val epochs = 30
    val batchSize = 10
    val learningRate = 3.0
    val lambda = 0.0
    val cost = CrossEntropyCost

    val nn = new Network(topology, cost)

    val trainingData = loadData("data/mnist_train.csv.gz").map {
      case (x, y) => (x, oneHotEncoded(y))
    }

    nn.sgd(trainingData, epochs, batchSize, learningRate, lambda)

  }

}
