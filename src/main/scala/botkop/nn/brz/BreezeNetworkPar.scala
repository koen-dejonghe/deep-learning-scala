package botkop.nn.brz

import akka.NotUsed
import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl.{Flow, Sink, Source}
import botkop.nn.Monitor
import breeze.linalg.{DenseMatrix, argmax}
import breeze.numerics.sigmoid

import scala.annotation.tailrec
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.language.postfixOps
import scala.util.Random

class BreezeNetworkPar(topology: List[Int]) {

  implicit val system = ActorSystem("QuickStart")
  implicit val materializer = ActorMaterializer()

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

  def backProp(x: DoubleMatrix, y: DoubleMatrix): (List[DoubleMatrix], List[DoubleMatrix]) = {

    val activations = feedForward(List(x))

    val deltaBias = activations.last - y
    val deltaWeight = deltaBias * activations(activations.size - 2).t

    val (nablaBiases, nablaWeights) = (topology.size - 2 until 0 by -1)
      .foldLeft((List(deltaBias), List(deltaWeight))) {
        case ((nbl, nwl), l) =>
          val sp = activations(l) *:* (-activations(l) + 1.0)
          val deltaBias = (weights(l).t * nbl.head) *:* sp
          val deltaWeight = deltaBias * activations(l - 1).t
          (deltaBias :: nbl, deltaWeight :: nwl)
      }

    (nablaBiases, nablaWeights)
  }

  def updateMiniBatch(miniBatch: List[(DoubleMatrix, DoubleMatrix)],
                      lm: Double,
                      lln: Double): Unit = {

    val zeroBiases: List[DoubleMatrix] = biases.map(b => DenseMatrix.zeros[Double](b.rows, b.cols))
    val zeroWeights: List[DoubleMatrix] = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

    miniBatch.foreach {
      case (x, y) =>
        val (deltaNablaB: List[DoubleMatrix], deltaNablaW: List[DoubleMatrix]) =
          backProp(x, y)

        zeroBiases.zip(deltaNablaB).foreach {
          case (nb, dnb) =>
            nb += dnb
        }

        zeroWeights.zip(deltaNablaW).foreach {
          case (nw, dnw) =>
            nw += dnw
        }
    }

    biases.zip(zeroBiases).foreach {
      case (b, nb) =>
        b -= nb * lm
    }

    weights.zip(zeroWeights).foreach {
      case (w, nw) =>
        w *= lln
        w -= nw * lm
    }
  }

  val bp: Flow[(DoubleMatrix, DoubleMatrix), (List[DoubleMatrix], List[DoubleMatrix]), NotUsed] =
    Flow[(DoubleMatrix, DoubleMatrix)]
      .mapAsyncUnordered(8) {
        case (x, y) => Future(backProp(x, y))
      }

  val inb: List[DoubleMatrix] = biases.map(b => DenseMatrix.zeros[Double](b.rows, b.cols))
  val inw: List[DoubleMatrix] = weights.map(w => DenseMatrix.zeros[Double](w.rows, w.cols))

  val add =
    Flow[(List[DoubleMatrix], List[DoubleMatrix])].fold(inb, inw) {
      case ((zbl, zwl), (nbl, nwl)) =>
        val b = zbl.zip(nbl).map {
          case (nb, dnb) => nb + dnb
        }

        val w = zwl.zip(nwl).map {
          case (nw, dnw) => nw + dnw
        }
        (b, w)
    }

  def updateMiniBatchPar(miniBatch: List[(DoubleMatrix, DoubleMatrix)],
                       lm: Double,
                       lln: Double): Unit = {

    val source = Source(miniBatch)

    val bw =
      Sink.fold[(List[DoubleMatrix], List[DoubleMatrix]), (List[DoubleMatrix], List[DoubleMatrix])](
        biases,
        weights) {
        case ((b, w), (nb, nw)) =>
          val ub = b.zip(nb).map { case (p1, p2) => p1 - (p2 * lm) }
          val uw = w.zip(nw).map { case (p1, p2) => (p1 * lln) - (p2 * lm) }
          (ub, uw)
      }

    val zzz: Future[(List[DoubleMatrix], List[DoubleMatrix])] =
      source.via(bp).via(add).runWith(bw)

    val (updb, updw) = Await.result(zzz, 5 seconds)

    biases = updb
    weights = updw

    // biases.zip(updb).foreach(v => v._1 :== v._2)
    // weights.zip(updw).foreach(v => v._1 :== v._2)

  }

  def accuracy(data: List[(DoubleMatrix, DoubleMatrix)]): Int = data.foldLeft(0) {
    case (r, (x, y)) =>
      val a = feedForward(List(x)).last
      val guess = argmax(a)
      val truth = argmax(y)
      if (guess == truth) r + 1 else r
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
//        updateMiniBatchPar(miniBatch, lm, lln)
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


