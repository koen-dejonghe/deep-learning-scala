package botkop.nn.akka

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import akka.actor.{ActorRef, ActorSystem}
import akka.pattern.ask
import akka.util.Timeout
import botkop.nn.akka.ActorNetwork.{dimensions, initialize, input, learningRate, numIterations, system, xTrain, yTrain}
import botkop.nn.akka.CostFunctions._
import botkop.nn.akka.gates._
import botkop.nn.akka.optimizers._
import numsca.Tensor

import scala.concurrent.duration._
import scala.io.Source
import scala.language.postfixOps

object MnistNetwork extends App {

  val system = ActorSystem()

  val (xTrain, yTrain) = loadData("data/mnist_train.csv.gz")
  // val (xTest, yTest) = loadData("data/mnist_test.csv.gz")

  val dimensions = Array(784, 50, 10)
  val learningRate = 0.01
  val numIterations = 5000
  val miniBatchSize = 10

  val (input, output) = initialize(dimensions, yTrain, learningRate)

  input ! Forward(xTrain, yTrain)

  def initialize(dimensions: Array[Int],
                 y: Tensor,
                 learningRate: Double): (ActorRef, ActorRef) = {

    val output = system.actorOf(OutputGate.props(softmaxLoss, numIterations))

    val (_, _, first) = dimensions.reverse
      .sliding(2)
      .foldLeft(true, dimensions.length - 1, output) {
        case ((isLast, i, next), shape) =>
          val nonLinearity =
            if (isLast)
              system.actorOf(SigmoidGate.props(next), s"sigmoid-gate-$i")
            else system.actorOf(ReluGate.props(next), s"relu-gate-$i")

          // val optimizer = GradientDescent(learningRate)
          //val optimizer = Adam(shape, learningRate)
          val optimizer = Momentum(shape, learningRate)

          (false,
           i - 1,
           system.actorOf(LinearGate.props(shape, nonLinearity, optimizer),
                          s"linear-gate-$i"))
      }

    val input =
      system.actorOf(InputGate.props(first, miniBatchSize = miniBatchSize))

    (input, output)

  }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String): (Tensor, Tensor) = {

    println(s"loading data from $fname: start")

    // val m = 1000

    val (xData, yData) = Source
      .fromInputStream(gzis(fname))
      .getLines()
      .take(100)
      .foldLeft(Array.empty[Double], Array.empty[Double]) {
        case ((xs, ys), line) =>
          val tokens = line.split(",")
          val (y, x) =
            (tokens.head.toDouble, tokens.tail.map(_.toDouble / 255.0))
          (xs ++ x, ys :+ y)
      }

    val x = Tensor(xData).reshape(yData.length, 784).transpose
    val y = Tensor(yData).reshape(yData.length, 1).transpose

    println(s"loading data from $fname: done")

    println(x.shape.toList)
    println(y.shape.toList)

    println(y)

    (x, y)

  }

}
