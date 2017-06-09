package botkop.nn

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.language.postfixOps

object TestPerf extends App {

  val topology = List(784, 30, 10)

  def f1(): Unit = {

    val t0 = System.currentTimeMillis()

    val biases: List[INDArray] =
      topology.tail.map(size => randn(size, 1))

    val weights: List[INDArray] =
      topology.sliding(2).map(t => randn(t(1), t.head)) toList

    val x = randn(784, 1)

    (1 to 1000000).foreach { i =>

      biases.zip(weights).foreach { case (b, w) =>

          b.transpose() dot w

      }

      /*
      biases.zip(weights).foldLeft(List(x)) {
        case (as, (b, w)) =>
           as :+ (w+w)
      }
      */

    }

    val t1 = System.currentTimeMillis()
    println(t1 - t0)
  }

  /*
  def f2(): Unit = {
    val t0 = System.currentTimeMillis()

    val biasShape = topology.tail :+ 1
    val biases = randn(biasShape.toArray)

    val weightShape =
    val weights: List[INDArray] =
      topology.sliding(2).map(t => randn(t(1), t.head)) toList


    val t1 = System.currentTimeMillis()
    println(t1 - t0)
  }
  */

  f1()

}
