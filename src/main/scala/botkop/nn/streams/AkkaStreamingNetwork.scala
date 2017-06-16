package botkop.nn.streams

import akka.NotUsed
import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import akka.stream.scaladsl.{Flow, Sink, Source}
import botkop.nn._
import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.annotation.tailrec
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.language.postfixOps
import scala.util.Random

class AkkaStreamingNetwork(topology: List[Int], cost: Cost) {

  implicit val system = ActorSystem("akka-streaming-neural-net")
  implicit val materializer = ActorMaterializer()

  // val (biases, weights) = initializeBiasesAndWeights(topology)

  def feedForward(x: Matrix,
                  biases: List[Matrix],
                  weights: List[Matrix]): List[Matrix] = {
    @tailrec
    def r(bws: List[(Matrix, Matrix)] = biases.zip(weights),
          acc: List[Matrix] = List(x)): List[Matrix] = bws match {
      case (b, w) :: rbws =>
        val z = (w dot acc.head) + b
        val a = sigmoid(z)
        r(rbws, a :: acc)
      case Nil =>
        acc
    }
    r().reverse
  }

  def backProp(x: Matrix,
               y: Matrix,
               biases: List[Matrix],
               weights: List[Matrix]): Future[(List[Matrix], List[Matrix])] = {

    val activations = feedForward(x, biases, weights)

    val delta = cost.delta(activations.last, y)

    val inb = delta
    val inw = delta dot activations(activations.size - 2).transpose()

    val source = Source(topology.size - 2 until 0 by -1)

    val sink = Sink
      .fold[(List[Matrix], List[Matrix]), Int]((List(inb), List(inw))) {
        case ((nbl, nwl), l) =>
          val sp = derivative(activations(l))

          // last added nb to nbl is the previous delta
          val delta = (weights(l).transpose() dot nbl.head) * sp

          val nb = delta
          val nw = delta dot activations(l - 1).transpose()

          (nb :: nbl, nw :: nwl)
      }

    source.runWith(sink)
  }

  /*
  def _updateMiniBatch(miniBatch: List[(Matrix, Matrix)],
                      lm: Double,
                      lln: Double): Unit = {

    lazy val inb: List[Matrix] = biases.map(b => zeros(b.shape(): _*))
    lazy val inw: List[Matrix] = weights.map(w => zeros(w.shape(): _*))

    val source = Source(miniBatch)

    val bp: Flow[(Matrix, Matrix), (List[Matrix], List[Matrix]), NotUsed] =
      Flow[(Matrix, Matrix)]
        .mapAsyncUnordered(2) {
          case (x, y) => backProp(x, y)
        }

    val add =
      Flow[(List[Matrix], List[Matrix])].fold(inb, inw) {
        case ((zbl, zwl), (nbl, nwl)) =>
          val b = zbl.zip(nbl).map {
            case (nb, dnb) => nb + dnb
          }

          val w = zwl.zip(nwl).map {
            case (nw, dnw) => nw + dnw
          }
          (b, w)
      }

    val bw =
      Sink.fold[(List[Matrix], List[Matrix]), (List[Matrix], List[Matrix])](
        biases,
        weights) {
        case ((b, w), (nb, nw)) =>
          val ub = b.zip(nb).map { case (p1, p2) => p1 - (p2 * lm) }
          val uw = w.zip(nw).map { case (p1, p2) => (p1 * lln) - (p2 * lm) }
          (ub, uw)
      }

    val zzz: Future[(List[Matrix], List[Matrix])] =
      source.via(bp).via(add).runWith(bw)

    val (updb, updw) = Await.result(zzz, 5 seconds)
    biases.zip(updb).foreach(v => v._1.assign(v._2))
    weights.zip(updw).foreach(v => v._1.assign(v._2))

  }
   */

  def updateMiniBatch(miniBatch: List[(Matrix, Matrix)],
                      biases: List[Matrix],
                      weights: List[Matrix],
                      lm: Double,
                      lln: Double): (List[Matrix], List[Matrix]) = {

    lazy val inb: List[Matrix] = biases.map(b => zeros(b.shape(): _*))
    lazy val inw: List[Matrix] = weights.map(w => zeros(w.shape(): _*))

    val source = Source(miniBatch)

    val bp: Flow[(Matrix, Matrix), (List[Matrix], List[Matrix]), NotUsed] =
      Flow[(Matrix, Matrix)]
        .mapAsyncUnordered(2) {
          case (x, y) => backProp(x, y, biases, weights)
        }

    val add =
      Flow[(List[Matrix], List[Matrix])].fold(inb, inw) {
        case ((zbl, zwl), (nbl, nwl)) =>
          val b = zbl.zip(nbl).map {
            case (nb, dnb) => nb + dnb
          }

          val w = zwl.zip(nwl).map {
            case (nw, dnw) => nw + dnw
          }
          (b, w)
      }

    val bw =
      Sink.fold[(List[Matrix], List[Matrix]), (List[Matrix], List[Matrix])](
        biases,
        weights) {
        case ((b, w), (nb, nw)) =>
          val ub = b.zip(nb).map { case (p1, p2) => p1 - (p2 * lm) }
          val uw = w.zip(nw).map { case (p1, p2) => (p1 * lln) - (p2 * lm) }
          (ub, uw)
      }

    val zzz: Future[(List[Matrix], List[Matrix])] =
      source.via(bp).via(add).runWith(bw)

    Await.result(zzz, 5 seconds)
  }

  def accuracy(data: List[(Matrix, Matrix)],
               biases: List[Matrix],
               weights: List[Matrix]): Int = data.foldLeft(0) {
    case (r, (x, y)) =>
      val a = feedForward(x, biases, weights).last
      val guess = argMax(a).getInt(0)
      val truth = argMax(y).getInt(0)
      if (guess == truth) r + 1 else r
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

    val (biases: List[Matrix], weights: List[Matrix]) =
      initializeBiasesAndWeights(topology)
    val lm = learningRate / miniBatchSize
    val lln = 1.0 - learningRate * (lambda / trainingData.size)

    (1 to epochs).foreach { epoch =>
      println(s"Epoch $epoch started")
      val t0 = System.currentTimeMillis()
      val shuffled = Random.shuffle(trainingData)
      val batches = shuffled.sliding(miniBatchSize, miniBatchSize)

      @tailrec
      def r(bts: List[List[(Matrix, Matrix)]],
            bas: List[Matrix],
            wes: List[Matrix]): (List[Matrix], List[Matrix]) = {
        bts match {
          case batch :: rbts =>
            val (b, w) = updateMiniBatch(batch, bas, wes, lm, lln)
            r(rbts, b, w)
          case Nil => (bas, wes)
        }
      }

      val (b, w) = r(batches.toList, biases, weights)

      /*
      shuffled.sliding(miniBatchSize, miniBatchSize).foreach { miniBatch =>
        updateMiniBatch(miniBatch, biases, weights, lm, lln)
      }
       */

      val t1 = System.currentTimeMillis()
      println(s"Epoch $epoch completed in ${t1 - t0} ms.")

      if (monitorEvaluationAccuracy) {
        val a = accuracy(evaluationData, b, w)
        println(s"Accuracy on evaluation data: $a / ${evaluationData.size}")
        monitor.evaluationAccuracy += a
      }

    }
    monitor
  }

}

object AkkaStreamingNetwork {

  def main(args: Array[String]) {
    val topology = List(784, 30, 30, 10)
    val epochs = 30
    val batchSize = 10
    val learningRate = 0.5
    val lambda = 0.5
    val cost = CrossEntropyCost

    val nn = new AkkaStreamingNetwork(topology, cost)

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
