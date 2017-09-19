package botkop.nn.akka

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import akka.actor.{ActorRef, ActorSystem}
import akka.pattern.ask
import akka.util.Timeout
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
  val (xTest, yTest) = loadData("data/mnist_test.csv.gz")

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String): (Tensor, Tensor) = {

    println(s"loading data from $fname: start")

    // val m = 1000

    val (xData, yData) = Source
      .fromInputStream(gzis(fname))
      .getLines()
      .take(10)
      .foldLeft(Array.empty[Double], Array.empty[Double]) { case ((xs, ys), line) =>
        val tokens = line.split(",")
        val (y, x) = (tokens.head.toDouble, tokens.tail.map(_.toDouble / 255.0))
        (xs ++ x, ys :+ y)
    }

    val y = Tensor(yData).reshape(yData.length, 1).transpose
    val x = Tensor(xData).reshape(yData.length, 784).transpose

    println(s"loading data from $fname: done")

    (x, y)
  }

}



