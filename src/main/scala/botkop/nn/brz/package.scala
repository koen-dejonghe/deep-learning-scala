package botkop.nn

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import breeze.generic.{MappingUFunc, UFunc}
import breeze.linalg.DenseMatrix
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
          // Xavier initialization
//          DenseMatrix.rand[Double](y, x, Rand.gaussian) /:/ Math.sqrt(y)
          // patch for relu
           DenseMatrix.rand[Double](y, x, Rand.gaussian) /:/ Math.sqrt(y / 2.0)
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

  import scala.{math => m}
  object reluPrime extends UFunc with MappingUFunc {
    implicit object relupDoubleImpl extends Impl[Double, Double] { def apply(v: Double): Double = m.max(0.0, v)}
    implicit object relupFloatImpl extends Impl[Float, Float] { def apply(v: Float): Float = m.max(0.0f, v)}
    implicit object relupabsIntImpl extends Impl[Int, Int] { def apply(v: Int): Int = m.max(0, v)}
    implicit object relupabsLongImpl extends Impl[Long, Long] { def apply(v: Long): Long = m.max(0L, v)}
  }

  object sigmoidPrime extends UFunc with MappingUFunc {
    implicit object spDoubleImpl extends Impl[Double, Double] { def apply(v: Double): Double = v * (1.0 - v) }
    implicit object spFloatImpl extends Impl[Float, Float] { def apply(v: Float): Float = v * (1.0f - v) }
    implicit object spIntImpl extends Impl[Int, Int] { def apply(v: Int): Int = v * (1 - v) }
    implicit object spLongImpl extends Impl[Long, Long] { def apply(v: Long): Long = v * (1L - v) }
  }

  def concat(data: List[DenseMatrix[Double]]): DenseMatrix[Double] = {
    require(data.nonEmpty)
    val numRows = data.size
    val numCols = data.head.rows
    val X = DenseMatrix.zeros[Double](numRows, numCols)
    for (i <- 0 until numRows) {
      val row = data(i).t
      for (j <- 0 until numCols) {
        val v: Double = row(0, j)
        X.update(i, j, v)
      }
    }
    X
  }



}
