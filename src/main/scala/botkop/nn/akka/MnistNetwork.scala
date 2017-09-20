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

  val system = ActorSystem()
  import system.dispatcher

  // implicit val executionContext = system.dispatchers.lookup("my-dispatcher")

  val dimensions = Array(784, 50, 10)
  val learningRate = 0.3
  val numIterations = 50000
  val miniBatchSize = 16
  val take = 100

  val (xTrain, yTrain) = loadData("data/mnist_train.csv.gz", take)
  val (xTest, yTest) = loadData("data/mnist_test.csv.gz", take)



  val (input, output) = initialize(dimensions, yTrain, learningRate)

  input ! Forward(xTrain, yTrain)


  implicit val timeout: Timeout = Timeout(5 seconds) // needed for `?`

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
    val p = numsca.argmax(x, 0)
    val acc = numsca.sum(p == y).squeeze() / m
    println(s"acc = $acc p = " + p + "\n           " + "y = " + y)
    acc
  }

  def initialize(dimensions: Array[Int],
                 y: Tensor,
                 learningRate: Double): (ActorRef, ActorRef) = {

    val output = system.actorOf(OutputGate.props(softmaxCost, numIterations))

    val (_, _, first) = dimensions.reverse
      .sliding(2)
      .foldLeft(true, dimensions.length - 1, output) {
        case ((isLast, i, next), shape) =>
          val nonLinearity =
            system.actorOf(ReluGate.props(next), s"relu-gate-$i")

          // val optimizer = GradientDescent(learningRate)
          val optimizer = Momentum(shape, learningRate)
          // val optimizer = Adam(shape, learningRate)

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

  def loadData(fname: String, take: Int): (Tensor, Tensor) = {

    println(s"loading data from $fname: start")

    val (xData, yData) = Source
      .fromInputStream(gzis(fname))
      .getLines()
      // .take(take)
      .foldLeft(List.empty[Double], List.empty[Double]) {
        case ((xs, ys), line) =>
          val tokens = line.split(",")
          val (y, x) =
            (tokens.head.toDouble, tokens.tail.map(_.toDouble / 255.0).toList)
          (x ::: xs, y :: ys)
      }

    println("done reading")

    val x = Tensor(xData.toArray).reshape(yData.length, 784).transpose
    val y = Tensor(yData.toArray).reshape(yData.length, 1).transpose

    println(s"loading data from $fname: done")

    println(x.shape.toList)
    println(y.shape.toList)

    println(y)

    (x, y)

  }

}
