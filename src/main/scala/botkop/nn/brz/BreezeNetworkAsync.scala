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
    : (List[DoubleMatrix], List[DoubleMatrix]) = {

    val activations = feedForward(List(x), bw.zipped)

    val deltaBias = activations.last - y
    val deltaWeight = deltaBias * activations(activations.size - 2).t

    @tailrec
    def r(l: Int,
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

  def backProp2(x: DoubleMatrix, y: DoubleMatrix)(
      implicit bw: BiasesAndWeights)
    : Future[(List[DoubleMatrix], List[DoubleMatrix])] = {

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
  }

  @tailrec
  final def feedForwardAsync(
      acc: List[DoubleMatrix],
      bws: List[(DoubleMatrix, DoubleMatrix)],
      y: DoubleMatrix): (List[DoubleMatrix], DoubleMatrix, DoubleMatrix) =
    bws match {
      case (b, w) :: rbws =>
        val z = (w * acc.head) + b
        val a = sigmoid(z)
        feedForwardAsync(a :: acc, rbws, y)
      case Nil =>
        val activations = acc.reverse
        val deltaBias = activations.last - y
        val deltaWeight = deltaBias * activations(activations.size - 2).t
        (activations, deltaBias, deltaWeight)
    }

  def backPropAsync(activations: List[DoubleMatrix],
                    deltaBias: DoubleMatrix,
                    deltaWeight: DoubleMatrix)(implicit bw: BiasesAndWeights)
    : (List[DoubleMatrix], List[DoubleMatrix]) = {

    @tailrec
    def r(l: Int,
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

  private def processEpoch(
      batches: List[List[(DoubleMatrix, DoubleMatrix)]],
      epoch: Int,
      startTime: Long)(implicit bw: BiasesAndWeights): Unit = {

    import bw._

    batches match {
      case miniBatch :: rbs =>
        for {
          /*
          adbdw <- Future.traverse(miniBatch) {
            case (x, y) => Future(feedForwardAsync(List(x), bw.zipped, y))
          }

          result <- Future.traverse(adbdw) {
            case (a, db, dw) => Future(backPropAsync(a, db, dw))
          }
           */

          result <- Future.traverse(miniBatch) {
            case (x, y) => backProp2(x, y)
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

          // update mini batch
          biases.zip(inb).foreach {
            case (b, nb) =>
              b -= nb * lm
          }

          weights.zip(inw).foreach {
            case (w, nw) =>
              w *= lln
              w -= nw * lm
          }

          processEpoch(rbs, epoch, startTime)
        }

      case Nil =>
        if (epoch > 0) {
          println(
            s"Epoch $epoch completed in ${System.currentTimeMillis() - startTime} ms.")
          /* accuracy */
          val a = accuracy(biases.zip(weights), evaluationData)
          println(s"Accuracy on evaluation data: $a / ${evaluationData.size}")
        }

        val shuffled = Random.shuffle(trainingData)
        val nextBatches = shuffled.sliding(miniBatchSize, miniBatchSize).toList
        processEpoch(nextBatches, epoch + 1, System.currentTimeMillis())

    }
  }

  @tailrec
  final def accuracy(bw: List[(DoubleMatrix, DoubleMatrix)],
                     data: List[(DoubleMatrix, DoubleMatrix)],
                     sum: Int = 0): Int =
    data match {
      case (x, y) :: ds =>
        val a = feedForward(List(x), bw).last
        val guess = argmax(a)
        val truth = argmax(y)
        accuracy(bw, ds, if (guess == truth) sum + 1 else sum)
      case Nil =>
        sum
    }

  def sgd(): Unit = {
    val (biases, weights) = initializeBiasesAndWeights(topology)
    processEpoch(Nil, 0, System.currentTimeMillis())(
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
