package botkop.nn.brz

import botkop.nn.Monitor
import breeze.linalg.{DenseMatrix, argmax}
import breeze.numerics.sigmoid

import scala.annotation.tailrec
import scala.concurrent.{Await, Future}
import scala.concurrent.duration._
import scala.language.postfixOps
import scala.util.Random

import scala.concurrent.ExecutionContext.Implicits.global

class BreezeNetwork(topology: List[Int]) {

  var (biases, weights) = initializeBiasesAndWeights(topology)

  @tailrec
  private def feedForward(acc: List[DoubleMatrix],
                          bws: List[(DoubleMatrix, DoubleMatrix)] =
                            biases.zip(weights)): List[DoubleMatrix] =
    bws match {
      case (b, w) :: rbws =>
        val z = (w * acc.head) + b
        val a = sigmoid(z)
        feedForward(a :: acc, rbws)
      case Nil =>
        acc.reverse
    }

  def backProp(x: DoubleMatrix,
               y: DoubleMatrix): (List[DoubleMatrix], List[DoubleMatrix]) = {

    val activations = feedForward(List(x))

    val deltaBias = activations.last - y
    val deltaWeight = deltaBias * activations(activations.size - 2).t

    @tailrec
    def r(l: Int,
          nbl: List[DoubleMatrix],
          nwl: List[DoubleMatrix]): (List[DoubleMatrix], List[DoubleMatrix]) =
      if (l > 0) {
        val sp = activations(l) *:* (-activations(l) + 1.0)
        val db = (weights(l).t * nbl.head) *:* sp
        val dw = db * activations(l - 1).t
        r(l - 1, db :: nbl, dw :: nwl)
      } else {
        (nbl, nwl)
      }

    r(topology.size - 2, List(deltaBias), List(deltaWeight))
  }

  // reference implementation!!!!!
  def collectDeltas(miniBatch: List[(DoubleMatrix, DoubleMatrix)])
    : (List[DoubleMatrix], List[DoubleMatrix]) = {

    val deltaBiases: List[DoubleMatrix] =
      biases.map(b => DenseMatrix.zeros[Double](b.rows, b.cols))
    val deltaWeights: List[DoubleMatrix] =
      weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

    miniBatch.foreach {
      case (x, y) =>
        val (deltaNablaB, deltaNablaW) = backProp(x, y)

        deltaBiases.zip(deltaNablaB).foreach {
          case (nb, dnb) =>
            nb += dnb
        }

        deltaWeights.zip(deltaNablaW).foreach {
          case (nw, dnw) =>
            nw += dnw
        }
    }
    (deltaBiases, deltaWeights)
  }

  def updateMiniBatch(miniBatch: List[(DoubleMatrix, DoubleMatrix)],
                      lm: Double,
                      lln: Double): Unit = {

    val (deltaBiases, deltaWeights) = collectDeltas(miniBatch)

    biases.zip(deltaBiases).foreach {
      case (b, nb) =>
        b -= nb * lm
    }

    weights.zip(deltaWeights).foreach {
      case (w, nw) =>
        w *= lln
        w -= nw * lm
    }
  }

  @tailrec
  final def accuracy(data: List[(DoubleMatrix, DoubleMatrix)],
                     sum: Int = 0): Int =
    data match {
      case (x, y) :: ds =>
        val a = feedForward(List(x)).last
        val guess = argmax(a)
        val truth = argmax(y)
        accuracy(ds, if (guess == truth) sum + 1 else sum)
      case Nil =>
        sum
    }

  def sgd(trainingData: List[(DoubleMatrix, DoubleMatrix)],
          epochs: Int,
          miniBatchSize: Int,
          learningRate: Double,
          lambda: Double,
          evaluationData: List[(DoubleMatrix, DoubleMatrix)] = List.empty,
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
        val a = accuracy(evaluationData)
        println(s"Accuracy on evaluation data: $a / ${evaluationData.size}")
        monitor.evaluationAccuracy += a
      }
    }

    monitor
  }
}

object BreezeNetwork {

  def main(args: Array[String]) {
    val topology = List(784, 100, 100, 10)
    val epochs = 30
    val batchSize = 100
    val learningRate = 1.5
    val lambda = 0.5

    val nn = new BreezeNetwork(topology)

    val (trainingData, validationData, testData) = mnistData()

    val monitor = nn.sgd(
      trainingData,
      epochs,
      batchSize,
      learningRate,
      lambda,
      testData,
      monitorEvaluationAccuracy = true
    )

    println(monitor)

  }
}
