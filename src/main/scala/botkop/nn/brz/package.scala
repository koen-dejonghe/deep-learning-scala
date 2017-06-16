package botkop.nn

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Rand

import scala.io.Source
import scala.language.postfixOps

package object brz {

  type DoubleMatrix = DenseMatrix[Double]

  def initializeBiasesAndWeights(
      topology: List[Int]): (List[DoubleMatrix], List[DoubleMatrix]) = {

    val biases =
      topology.tail.map(size => DenseMatrix.rand[Double](size, 1, Rand.gaussian))

    val weights = topology
      .sliding(2)
      .map(t => (t.head, t(1)))
      .map {
        case (x, y) =>
          DenseMatrix.rand[Double](y, x, Rand.gaussian) /:/ Math.sqrt(x)
      }
      .toList

    (biases, weights)
  }

  def vectorize(x: Int, base: Int = 10): DoubleMatrix = {
    val m = DenseMatrix.zeros[Double](base, 1)
    m(x to x, 0 to 0) := 1.0
    m
  }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String): List[(DoubleMatrix, Int)] = {
    Source.fromInputStream(gzis(fname)).getLines() map { line =>
      val tokens = line.split(",")
      val (y, x) = (tokens.head.toInt, tokens.tail.map(_.toDouble / 255.0))
      (new DenseMatrix(x.length, 1, x), y)
    } toList
  }

  def mnistData(): (List[(DoubleMatrix, DoubleMatrix)],
    List[(DoubleMatrix, DoubleMatrix)],
    List[(DoubleMatrix, DoubleMatrix)]) = {

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

}
