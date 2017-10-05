package botkop.nn.akka

import java.io._
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

  // import org.nd4j.linalg.api.buffer.DataBuffer
  // import org.nd4j.linalg.api.buffer.util.DataTypeUtil
  // DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  val system: ActorSystem = ActorSystem()
  import system.dispatcher

  // the optimizer is an object that contains state,
  // so must be recreated for each gate
  // hence this is a function call rather than a value
  // def optimizer() = Nesterov(learningRate = 0.1)
  def optimizer() = Adam(learningRate = 0.0001)

  // def optimizer() = GradientDescent(learningRate = 0.3)

  /*
  val layout = (Linear + Relu) * 2
  val network = layout
    .withDimensions(784, 50, 10)
    .withOptimizer(optimizer)
    .withCostFunction(softmaxCost)
    .withRegularization(1e-4)
    .withMiniBatchSize(16)

  val (input, output) = network.init()

   */

  val (input, output) =
    ((Linear + Relu + Dropout) * 2 + Linear)
    // ((Linear + Relu) * 2)
      .withDimensions(784, 100, 50, 10)
      .withOptimizer(optimizer)
      .withCostFunction(softmaxCost)
      .withRegularization(1e-5)
      .withMiniBatchSize(64)
      .init()

  val take = Some(100)
  // val take = None

  val (xTrain, yTrain) = loadData("data/mnist_train.csv.gz", take)
  val (xDev, yDev) = loadData("data/mnist_test.csv.gz", take)

  val xTest =
    readData("data/mnist-kaggle-test.csv", Array(28000, 784)).transpose

  input ! Forward(xTrain, yTrain)

  monitor()

  implicit val timeout: Timeout = Timeout(1 second) // needed for `?`

  while(true) {
    scala.io.StdIn.readLine()

    println("writing test result")

    (input ? Predict(xTest)).mapTo[Tensor].onComplete { d =>
      val p = numsca.argmax(d.get, 0)
      val writer = new BufferedWriter(
        new OutputStreamWriter(
          new FileOutputStream("kaggle-mnist-submission.csv")))

      writer.write(s"ImageId,Label\n")

      p.data.zipWithIndex.foreach { case (n, i) =>
        writer.write(s"${i + 1},${n.toInt}\n")
      }

      writer.close()
    }
  }

  def monitor() = system.scheduler.schedule(0 seconds, 5 seconds) {

    (input ? Predict(xDev)).mapTo[Tensor].onComplete { d =>
      logger.info(s"accuracy on test set: ${accuracy(d.get, yDev)}")
    }

    (input ? Predict(xTrain)).mapTo[Tensor].onComplete { d =>
      val (c, _) = softmaxCost(d.get, yTrain)
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
      .foldLeft(List.empty[Float], List.empty[Float]) {
        case ((xs, ys), line) =>
          val tokens = line.split(",")
          val (y, x) =
            (tokens.head.toFloat, tokens.tail.map(_.toFloat / 255).toList)
          (x ::: xs, y :: ys)
      }

    val x = Tensor(xData.toArray).reshape(yData.length, 784).transpose
    val y = Tensor(yData.toArray).reshape(yData.length, 1).transpose

    logger.info(s"loading data from $fname: done")

    (x, y)

  }

  def readData(fileName: String, shape: Array[Int]): Tensor = {
    val data = Source
      .fromFile(fileName)
      .getLines()
      .map(_.split(",").map(_.toFloat / 255))
      .flatten
      .toArray
    Tensor(data).reshape(shape)
  }

}
