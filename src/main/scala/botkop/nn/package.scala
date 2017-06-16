package botkop
import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4s.Implicits._

import scala.io.Source
import scala.language.postfixOps

package object nn {

  def derivative(z: Matrix): Matrix = z * (-z + 1.0)

  def vectorize(x: Int, base: Int = 10): Matrix =
    zeros(1, base) putScalar (x, 1.0)

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String): List[(Matrix, Int)] = {
    Source.fromInputStream(gzis(fname)).getLines() map { line =>
      val tokens = line.split(",")
      val (y, x) = (tokens.head.toInt, tokens.tail.map(_.toDouble / 255.0))
      (create(x).transpose(), y)
    } toList
  }

  def mnistData(): (List[(Matrix, Matrix)],
    List[(Matrix, Matrix)],
    List[(Matrix, Matrix)]) = {

    println("reading training data")
    val tvData = loadData("data/mnist_train.csv.gz").map {
      case (x, y) => (x, vectorize(y))
    }
    println("done reading training data")

    val (trainingData, validationData) = tvData.splitAt(50000)

    println("reading test data")
    val testData = loadData("data/mnist_test.csv.gz").map {
      case (x, y) => (x, vectorize(y))
    }
    println("done reading test data")

    (trainingData, validationData, testData)
  }

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
  def initializeBiasesAndWeights(topology: List[Int]): (List[Matrix], List[Matrix]) = {

    val biases: List[Matrix] =
      topology.tail.map(size => randn(size, 1))

    val weights: List[Matrix] = topology
      .sliding(2)
      .map(t => (t.head, t(1)))
      .map { case (x, y) => randn(y, x) / Math.sqrt(x) }
      .toList

    (biases, weights)
  }

}
