package botkop.nn.brz

import breeze.linalg.{DenseMatrix, argmax}
import breeze.numerics.sigmoid

import scala.annotation.tailrec
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.language.postfixOps
import scala.util.Random

class BreezeNetworkAsync(topology: List[Int],
                         trainingData: List[(DoubleMatrix, DoubleMatrix)],
                         epochs: Int,
                         miniBatchSize: Int,
                         learningRate: Double,
                         lambda: Double,
                         evaluationData: List[(DoubleMatrix, DoubleMatrix)] =
                           List.empty,
                         monitorEvaluationCost: Boolean = false,
                         monitorEvaluationAccuracy: Boolean = false,
                         monitorTrainingCost: Boolean = false,
                         monitorTrainingAccuracy: Boolean = false) {

  val lm: Double = learningRate / miniBatchSize
  val lln: Double = 1.0 - learningRate * (lambda / trainingData.size)

  case class BiasesAndWeights(biases: List[DoubleMatrix],
                              weights: List[DoubleMatrix]) {
    lazy val zipped: List[(DoubleMatrix, DoubleMatrix)] = biases.zip(weights)
  }

  @tailrec
  private def feedForward(
      acc: List[DoubleMatrix],
      bws: List[(DoubleMatrix, DoubleMatrix)]): List[DoubleMatrix] =
    bws match {
      case (b, w) :: rbws =>
        val z = (w * acc.head) + b
        val a = sigmoid(z)
        feedForward(a :: acc, rbws)
      case Nil =>
        acc.reverse
    }

  def backProp(x: DoubleMatrix, y: DoubleMatrix)(implicit bw: BiasesAndWeights)
    : Future[(List[DoubleMatrix], List[DoubleMatrix])] =
    for {
      activations <- Future(feedForward(List(x), bw.zipped))
    } yield {

      val deltaBias = activations.last - y
      val deltaWeight = deltaBias * activations(activations.size - 2).t

      @tailrec
      def r(
          l: Int,
          nbl: List[DoubleMatrix],
          nwl: List[DoubleMatrix]): (List[DoubleMatrix], List[DoubleMatrix]) =
        if (l > 0) {
          val sp = activations(l) *:* (-activations(l) + 1.0)
          val db = (bw.weights(l).t * nbl.head) *:* sp
          val dw = db * activations(l - 1).t
          r(l - 1, db :: nbl, dw :: nwl)
        } else {
          (nbl, nwl)
        }

      r(topology.size - 2, List(deltaBias), List(deltaWeight))
    }

  def updateMiniBatch(
      miniBatch: List[(DoubleMatrix, DoubleMatrix)],
      remainingBatches: List[List[(DoubleMatrix, DoubleMatrix)]])(
      implicit epoch: Int,
      startTime: Long,
      bw: BiasesAndWeights): Unit = {

    import bw._

    for {
      result <- Future.traverse(miniBatch) {
        case (x, y) => backProp(x, y)
      }
    } yield {

      // aggregate nablas
      val inb: List[DoubleMatrix] =
        biases.map(b => DenseMatrix.zeros[Double](b.rows, b.cols))
      val inw: List[DoubleMatrix] =
        weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

      result.foreach {
        case (b, w) =>
          inb.zip(b).foreach { case (p1, p2) => p1 += p2 }
          inw.zip(w).foreach { case (p1, p2) => p1 += p2 }
      }

      // update biases
      biases.zip(inb).foreach {
        case (b, nb) =>
          b -= nb * lm
      }

      // update weights
      weights.zip(inw).foreach {
        case (w, nw) =>
          w *= lln
          w -= nw * lm
      }

      // call processEpoch with the remainder of the batches
      processEpoch(remainingBatches)
    }

  }

  private def processEpoch(batches: List[List[(DoubleMatrix, DoubleMatrix)]])(
      implicit epoch: Int,
      startTime: Long,
      bw: BiasesAndWeights): Unit = batches match {

    case miniBatch :: rbs =>
      updateMiniBatch(miniBatch, rbs)

    case Nil =>
      if (epoch > 0) {
        println(
          s"Epoch $epoch completed in ${System.currentTimeMillis() - startTime} ms.")
        accuracy(evaluationData)
      }

      val shuffled = Random.shuffle(trainingData)
      val nextBatches = shuffled.sliding(miniBatchSize, miniBatchSize).toList
      processEpoch(nextBatches)(epoch + 1, System.currentTimeMillis(), bw)
  }

  def accuracy(data: List[(DoubleMatrix, DoubleMatrix)], sum: Int = 0)(
      implicit bw: BiasesAndWeights): Unit =
    for {
      correct <- Future.traverse(data) {
        case (x, y) =>
          Future {
            val a = feedForward(List(x), bw.zipped).last
            val guess = argmax(a)
            val truth = argmax(y)
            if (guess == truth) 1 else 0
          }
      }
    } yield {
      val c = correct.sum
      println(s"Accuracy on evaluation data: $c / ${data.size}")
    }

  def sgd(): Unit = {
    val (biases, weights) = initializeBiasesAndWeights(topology)
    processEpoch(Nil)(0,
                      System.currentTimeMillis(),
                      BiasesAndWeights(biases, weights))
  }
}

object BreezeNetworkAsync {

  def main(args: Array[String]) {
    val topology = List(784, 100, 100, 10)
    val epochs = 30
    val batchSize = 100
    val learningRate = 1.5
    val lambda = 0.5

    val (trainingData, validationData, testData) = mnistData()

    val nn = new BreezeNetworkAsync(topology,
                                    trainingData,
                                    epochs,
                                    batchSize,
                                    learningRate,
                                    lambda,
                                    testData,
                                    monitorEvaluationAccuracy = true)

    nn.sgd()

    Thread.sleep(Long.MaxValue)
  }

}
