package botkop.nn.actors

import akka.actor.ActorSystem
import akka.pattern.ask
import akka.util.Timeout
import botkop.nn.Network.mnistData
import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._

import scala.concurrent.Await
import scala.concurrent.duration._
import scala.io.StdIn
import scala.language.postfixOps
import scala.util.Random

object NetworkApp extends App {

  val system: ActorSystem = ActorSystem()

  implicit val timeout = Timeout(10 seconds)

  val hyperParameters = HyperParameters(
    miniBatchSize = 10,
    learningRate = 0.3,
    lambda = 0.5,
    trainingDataSize = 50000
  )

  val hiddenLayer =
    system.actorOf(Layer.props(Shape(784, 30), hyperParameters))

  val outputLayer =
    system.actorOf(Layer.props(Shape(30, 10), hyperParameters))

  hiddenLayer ! Wiring(Some(outputLayer), None)
  outputLayer ! Wiring(None, Some(hiddenLayer))

  val miniBatchActor = system.actorOf(MiniBatchActor.props(hiddenLayer))

  val (trainingData, validationData, testData) = mnistData()

//  sgd(trainingData.take(5000), 30, hyperParameters.miniBatchSize, testData.take(1000))
  sgd(trainingData, 30, hyperParameters.miniBatchSize, testData)

  def updateMiniBatch(miniBatch: List[(Matrix, Matrix)]): Unit = {
    val f = miniBatchActor ? MiniBatch(miniBatch)
    Await.result(f, timeout.duration)
  }

  def accuracy(data: List[(Matrix, Matrix)]): Int = data.foldLeft(0) {
    case (r, (x, y)) =>
      val f = hiddenLayer ? Guess(x)
      val a = Await.result(f, timeout.duration).asInstanceOf[Matrix]

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

      val a = accuracy(evaluationData)
      println(s"Accuracy on evaluation data: $a / ${evaluationData.size}")
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

