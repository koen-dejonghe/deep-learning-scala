package botkop.nn.akka

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import akka.actor.{ActorRef, ActorSystem}
import akka.util.Timeout
import botkop.nn.akka.CostFunctions._
import botkop.nn.akka.gates._
import botkop.nn.akka.optimizers._
import numsca.Tensor

import scala.concurrent.duration._
import akka.pattern.ask

import scala.io.Source
import scala.language.postfixOps

object MnistNetwork extends App {

  import org.nd4j.linalg.api.buffer.DataBuffer
  import org.nd4j.linalg.api.buffer.util.DataTypeUtil

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  val system = ActorSystem()
  import system.dispatcher

  val dimensions = Array(784, 100, 10)
  // val learningRate = 0.3

  // def optimizer = Adam(0.001)
  def optimizer = Momentum(0.3)

  val regularization = 1e-4
  // val regularization = 0.0
  val numIterations = 500000
  val miniBatchSize = 16
  val take = Some(1000)
  // val take = None

  val (xTrain, yTrain) = loadData("data/mnist_train.csv.gz", take)
  val (xTest, yTest) = loadData("data/mnist_test.csv.gz", take)

  val (input, output) = initialize(dimensions, regularization, optimizer)

  input ! Forward(xTrain, yTrain)

  implicit val timeout: Timeout = Timeout(5 seconds) // needed for `?`

  while (true) {
    Thread.sleep(5000)

    val ta = (input ? Predict(xTest)).mapTo[Tensor]
    ta.onComplete { d =>
      println(s"accuracy on test set: ${accuracy(d.get, yTest)}")
    }

    val tra = (input ? Predict(xTrain)).mapTo[Tensor]
    tra.onComplete { d =>
      val (cost, _) = softmaxCost(d.get, yTrain)
      println(
        s"accuracy on training set: ${accuracy(d.get, yTrain)} cost: $cost")
    }
  }

  def accuracy(x: Tensor, y: Tensor): Double = {
    val m = x.shape(1)
    val p = numsca.argmax(x, 0)
    val acc = numsca.sum(p == y).squeeze() / m
    acc
  }

  def initialize(dimensions: Array[Int],
                 regularization: Double,
                 optimizer: => Optimizer): (ActorRef, ActorRef) = {

    val output = system.actorOf(OutputGate.props(softmaxCost, numIterations))

    val (_, first) = dimensions.reverse
      .sliding(2)
      .foldLeft(dimensions.length - 1, output) {
        case ((i, next), shape) =>
          val nonLinearity =
            system.actorOf(ReluGate.props(next), s"relu-gate-$i")

          val linearity =
            system.actorOf(
              LinearGate.props(shape, nonLinearity, regularization, optimizer),
              s"linear-gate-$i")

          (i - 1, linearity)
      }

    val input =
      system.actorOf(InputGate.props(first, miniBatchSize = miniBatchSize))

    (input, output)

  }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String, take: Option[Int] = None): (Tensor, Tensor) = {

    println(s"loading data from $fname: start")

    val lines = take match {
      case Some(n) =>
        Source
          .fromInputStream(gzis(fname))
          .getLines()
          .take(n)
      case None =>
        Source
          .fromInputStream(gzis(fname))
          .getLines()
    }

    val (xData, yData) = lines
      .foldLeft(List.empty[Double], List.empty[Double]) {
        case ((xs, ys), line) =>
          val tokens = line.split(",")
          val (y, x) =
            (tokens.head.toDouble, tokens.tail.map(_.toDouble / 255.0).toList)
          (x ::: xs, y :: ys)
      }

    val x = Tensor(xData.toArray).reshape(yData.length, 784).transpose
    val y = Tensor(yData.toArray).reshape(yData.length, 1).transpose

    println(s"loading data from $fname: done")

    (x, y)

  }

}
