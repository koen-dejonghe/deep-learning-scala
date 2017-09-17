package botkop.nn.akka

import akka.actor.{ActorRef, ActorSystem}
import akka.pattern.ask
import akka.util.Timeout
import botkop.nn.akka.CostFunctions._
import botkop.nn.akka.gates._
import botkop.nn.akka.optimizers.{Adam, GradientDescent}
import numsca.Tensor

import scala.concurrent.duration._
import scala.io.Source
import scala.language.postfixOps

object ActorNetwork extends App {

  val system = ActorSystem()

  val xTrain =
    readData("data/coursera/catvsnoncat/train_x.csv", Array(12288, 209))
  val yTrain =
    readData("data/coursera/catvsnoncat/train_y.csv", Array(1, 209))

  val xTest =
    readData("data/coursera/catvsnoncat/test_x.csv", Array(12288, 50))
  val yTest =
    readData("data/coursera/catvsnoncat/test_y.csv", Array(1, 50))

  val dimensions = Array(12288, 20, 7, 5, 1)
  val learningRate = 0.0075
  // val learningRate = 0.03
  val numIterations = 2500

  val (input, output) = initialize(dimensions, yTrain, learningRate)

  input ! Forward(xTrain)

  implicit val timeout: Timeout = Timeout(5 seconds) // needed for `?`
  import system.dispatcher

  while (true) {
    Thread.sleep(3000)

    val ta = (input ? Predict(xTest)).mapTo[Tensor]
    ta.onComplete { d =>
      println(s"accuracy on test set: ${accuracy(d.get, yTest)}")
    }

    val tra = (input ? Predict(xTrain)).mapTo[Tensor]
    tra.onComplete { d =>
      println(s"accuracy on training set: ${accuracy(d.get, yTrain)}")
    }

  }

  def accuracy(x: Tensor, y: Tensor): Double = {
    val m = x.shape(1)
    val p = x > 0.5
    numsca.sum(p == y).squeeze() / m
  }

  def initialize(dimensions: Array[Int],
                 y: Tensor,
                 learningRate: Double): (ActorRef, ActorRef) = {

    val output = system.actorOf(
      OutputGate.props(y, crossEntropyCost, numIterations))

    val (_, first) = dimensions.reverse.sliding(2).foldLeft(true, output) {
      case ((isLast, next), shape) =>
        val nonLinearity =
          if (isLast) system.actorOf(SigmoidGate.props(next))
          else system.actorOf(ReluGate.props(next))

        // val optimizer = GradientDescent(learningRate)
        val optimizer = Adam(shape, learningRate)

        (false,
         // system.actorOf(LinearGate.props(shape, nonLinearity, learningRate))
         system.actorOf(LinearGate.props(shape, nonLinearity, optimizer))
        )
    }

    val input = system.actorOf(InputGate.props(first))

    (input, output)
  }

  def readData(fileName: String, shape: Array[Int]): Tensor = {
    val data = Source
      .fromFile(fileName)
      .getLines()
      .map(_.split(",").map(_.toDouble))
      .flatten
      .toArray
    Tensor(data).reshape(shape)
  }
}



