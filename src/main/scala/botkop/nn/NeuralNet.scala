package botkop.nn

import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.annotation.tailrec
import scala.language.postfixOps
import scala.util.Random

class NeuralNet(topology: List[Int], cost: Cost) {

  val (biases, weights) = initializeBiasesAndWeights(topology)

  /**
    * Return the output of the network if x is input
    * @param x input to the network
    * @return all activated layers
    */
  def _feedForward(x: Matrix): List[Matrix] = {
    biases.zip(weights).foldLeft(List(x)) {
      case (as, (b, w)) =>
        val z = (w dot as.last) + b
        val a = sigmoid(z)
        as :+ a
    }
  }

  @tailrec
  private def feedForward(
      acc: List[Matrix],
      bws: List[(Matrix, Matrix)] = biases.zip(weights)): List[Matrix] =
    bws match {
      case (b, w) :: rbws =>
        val z = (w dot acc.head) + b
        val a = sigmoid(z)
        feedForward(a :: acc, rbws)
      case Nil =>
        acc.reverse
    }

  def feedForward(x: Matrix): List[Matrix] = feedForward(List(x))

  def backProp(x: Matrix, y: Matrix): (List[Matrix], List[Matrix]) = {

    val activations = feedForward(List(x))

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
                      lm: Double,
                      lln: Double): Unit = {

    val nablaBiases: List[Matrix] = biases.map(b => zeros(b.shape(): _*))
    val nablaWeights: List[Matrix] = weights.map(w => zeros(w.shape(): _*))

    miniBatch.foreach {
      case (x, y) =>
        val (deltaNablaB: List[Matrix], deltaNablaW: List[Matrix]) =
          backProp(x, y)

        nablaBiases.zip(deltaNablaB).foreach {
          case (nb, dnb) =>
            nb += dnb
        }

        nablaWeights.zip(deltaNablaW).foreach {
          case (nw, dnw) =>
            nw += dnw
        }
    }

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

  def totalCost(data: List[(Matrix, Matrix)], lambda: Double): Double = {

    val ffCost: Double = data.foldLeft(0.0) {
      case (c, (x, y)) =>
        val a = feedForward(x).last
        c + cost.function(a, y) / data.size
    }

    weights.foldLeft(ffCost) {
      case (c, w) =>
        c + .5 * (lambda / data.size) * w.norm2Number().doubleValue()
    }
  }

  def accuracy(data: List[(Matrix, Matrix)]): Int = data.foldLeft(0) {
    case (r, (x, y)) =>
      val a = feedForward(x).last
      val guess = argMax(a).getInt(0)
      val truth = argMax(y).getInt(0)
      if (guess == truth) r + 1 else r
  }

  /**
    *
    * Train the neural network using mini-batch stochastic gradient
    * descent.  The trainingData is a list of tuples (x, y)
    * representing the training inputs and the desired outputs.
    * The method also accepts
    * evaluation_data, usually either the validation or test
    * data.  We can monitor the cost and accuracy on either the
    * evaluation data or the training data, by setting the
    * appropriate flags.  The method returns a tuple containing four
    * lists: the (per-epoch) costs on the evaluation data, the
    * accuracies on the evaluation data, the costs on the training
    * data, and the accuracies on the training data.  All values are
    * evaluated at the end of each training epoch.  So, for example,
    * if we train for 30 epochs, then the first element of the tuple
    * will be a 30-element list containing the cost on the
    * evaluation data at the end of each epoch. Note that the lists
    * are empty if the corresponding flag is not set.
    */
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

      if (monitorTrainingCost) {
        val c = totalCost(trainingData, lambda)
        println(s"Cost on training data: $c")
        monitor.trainingCost += c
      }

      if (monitorTrainingAccuracy) {
        val a = accuracy(trainingData)
        println(s"Accuracy on training data: $a / ${trainingData.size}")
        monitor.trainingAccuracy += a
      }

      if (monitorEvaluationCost) {
        val c = totalCost(evaluationData, lambda)
        println(s"Cost on evaluation data: $c")
        monitor.evaluationCost += c
      }

      if (monitorEvaluationAccuracy) {
        val a = accuracy(evaluationData)
        println(s"Accuracy on evaluation data: $a / ${evaluationData.size}")
        monitor.evaluationAccuracy += a
      }
    }
    monitor
  }
}

object NeuralNet {

  def main(args: Array[String]) {
    val topology = List(784, 30, 30, 10)
    val epochs = 30
    val batchSize = 10
    val learningRate = 0.5
    val lambda = 0.5
    val cost = CrossEntropyCost

    val nn = new NeuralNet(topology, cost)

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
