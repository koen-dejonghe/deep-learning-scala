package botkop.nn.akka

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import akka.actor.ActorSystem
import akka.pattern.ask
import akka.util.Timeout
import botkop.nn.akka.CostFunctions._
import botkop.nn.akka.gates._
import botkop.nn.akka.optimizers._
import com.typesafe.scalalogging.LazyLogging
import numsca.Tensor

import scala.concurrent.duration._
import scala.io.Source
import scala.language.postfixOps

object MnistNetwork extends App with LazyLogging {

  import org.nd4j.linalg.api.buffer.DataBuffer
  import org.nd4j.linalg.api.buffer.util.DataTypeUtil

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  val system: ActorSystem = ActorSystem()
  import system.dispatcher

  val layout = (Linear + Relu) * 2
  val dimensions = Array(784, 50, 10)

  // def optimizer = Adam(learningRate = 0.001)
  def optimizer = Momentum(learningRate = 0.3)

  val cost: CostFunction = softmaxCost

  val regularization = 1e-4
  // val regularization = 0.0

  val miniBatchSize = 16

  val (input, output) = Network.initialize(layout,
                                           dimensions,
                                           miniBatchSize,
                                           cost,
                                           optimizer,
                                           regularization)

  val take = Some(1000)
  // val take = None

  val (xTrain, yTrain) = loadData("data/mnist_train.csv.gz", take)
  val (xTest, yTest) = loadData("data/mnist_test.csv.gz", take)

  input ! Forward(xTrain, yTrain)

  monitor()

  def monitor() = system.scheduler.schedule(5 seconds, 5 seconds) {

    implicit val timeout: Timeout = Timeout(1 second) // needed for `?`

    (input ? Predict(xTest)).mapTo[Tensor].onComplete { d =>
      logger.info(s"accuracy on test set: ${accuracy(d.get, yTest)}")
    }

    (input ? Predict(xTrain)).mapTo[Tensor].onComplete { d =>
      val (c, _) = cost(d.get, yTrain)
      val a = accuracy(d.get, yTrain)
      logger.info(s"accuracy on training set: $a cost: $c")
    }
  }

  def accuracy(x: Tensor, y: Tensor): Double = {
    val m = x.shape(1)
    val p = numsca.argmax(x, 0)
    val acc = numsca.sum(p == y) / m
    acc
  }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String, take: Option[Int] = None): (Tensor, Tensor) = {

    logger.info(s"loading data from $fname: start")

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

    logger.info(s"loading data from $fname: done")

    (x, y)

  }

}
