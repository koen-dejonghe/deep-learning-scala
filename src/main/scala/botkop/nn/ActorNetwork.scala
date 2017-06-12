package botkop.nn

import akka.actor.{Actor, ActorSystem, Props}
import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.util.Random

object ActorNetwork extends App {

  val system: ActorSystem = ActorSystem()

  val hyperParameters = HyperParameters(
    miniBatchSize = 10,
    learningRate = 0.5,
    lambda = 0.5,
    trainingDataSize = 50000
  )

  val hiddenLayer =
    system.actorOf(Layer.props(Shape(784, 30), hyperParameters))

  val outputLayer =
    system.actorOf(Layer.props(Shape(30, 10), hyperParameters))

  val collector = system.actorOf(Collector.props())

  hiddenLayer ! Wiring(None, Some(outputLayer))
  outputLayer ! Wiring(Some(hiddenLayer), None)

  def updateMiniBatch(miniBatch: List[(Matrix, Matrix)]): Unit = {
    miniBatch.foreach { case (x, y) =>
        hiddenLayer ! FeedForward(x, y)
    }
    hiddenLayer ! UpdateWeightsAndBiases
  }

  def accuracy(data: List[(Matrix, Matrix)]): Int = data.foldLeft(0) {
    case (r, (x, y)) =>
      val a =
      val guess = argMax(a).getInt(0)

      val truth = argMax(y).getInt(0)
      if (guess == truth) r + 1 else r
  }

  def sgd(trainingData: List[(Matrix, Matrix)],
          epochs: Int,
          miniBatchSize: Int,
          evaluationData: List[(Matrix, Matrix)] = List.empty): Unit = {

    (1 to epochs).foreach { epoch =>
      val t0 = System.currentTimeMillis()
      val shuffled = Random.shuffle(trainingData)
      shuffled.sliding(miniBatchSize, miniBatchSize).foreach { miniBatch =>
        updateMiniBatch(miniBatch)
      }
      val t1 = System.currentTimeMillis()
      println(s"Epoch $epoch completed in ${t1 - t0} ms.")

      val a = accuracy(trainingData)
      println(s"Accuracy on training data: $a / ${trainingData.size}")
    }
  }

}

case class HyperParameters(miniBatchSize: Int,
                           learningRate: Double,
                           lambda: Double,
                           trainingDataSize: Int) {

  val lm: Double = learningRate / miniBatchSize // todo: this can cause problems when trainingDataSize % miniBatchSize != 0
  val lln: Double = 1.0 - learningRate * (lambda / trainingDataSize)
}

class Collector extends Actor {
  override def receive: Receive = {
    case o =>
      println(o)
  }
}

object Collector {
  def props(): Props = Props(new Collector())
}
