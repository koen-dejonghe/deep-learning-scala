package botkop.nn.cs231n

import botkop.nn.brz._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand

import scala.io.Source

object SoftmaxLinearClassifier extends App {

  val N = 100 // number of points per class
  val D = 2 // dimensionality
  val K = 3 // number of classes

  val data = readData()
  val X: DenseMatrix[Double] = data._1
  val yv: DenseMatrix[Double] = data._2
  val y = data._3

  // initialize parameters randomly
  val W = DenseMatrix.rand(D, K, Rand.gaussian) * 0.01
  val b = DenseVector.zeros[Double](K)

  val step_size = 1e-0
  val reg: Double = 1e-3 // regularization strength

  val num_examples = X.rows.toDouble

  for (i <- 1 to 1000) {

    // compute class scores for a linear classifier
    val scores: DenseMatrix[Double] = X * W // + b
    scores(*, ::) :+= b

    // get unnormalized probabilities
    val exp_scores: DenseMatrix[Double] = exp(scores) //300x3
    // normalize them for each example
    val s: DenseVector[Double] = sum(exp_scores(*,::)) //300x1
    val probs: DenseMatrix[Double] = exp_scores(::,*) / s

    //gradient on the scores
    // val dscores: DenseMatrix[Double] = (probs - y) /:/ num_examples
    val dscores: DenseMatrix[Double] = probs
    dscores(::, y.toScalaVector()) :-= 1.0
    dscores :/= num_examples

    // backpropate the gradient to the parameters (W,b)
    val dW = X.t * dscores
//    val db: DenseVector[Double] = sum(dscores, Axis._0).inner
    val db: DenseVector[Double] = sum(dscores(::,*)).t

    dW :+= W * reg // regularization gradient

    // perform a parameter update
    W :+= dW * -step_size
    b :+= db * -step_size

  }

  val scores = X * W
  scores(*, ::) :+= b

  val predicted_class = argmax(scores, Axis._1).toDenseMatrix
  val yp = argmax(yv, Axis._1).toDenseMatrix
  val result = predicted_class :== yp

  val num_correct = predicted_class.data.zip(yp.data).count{t =>
    t._1 == t._2
  }
  println(num_correct / num_examples)


  def printShape(s: String, d: DenseMatrix[Double]): Unit =
    println(s"$s: ${d.rows}x${d.cols}")

  def readData(): (DenseMatrix[Double], DenseMatrix[Double], DenseVector[Int]) = {
    val (xArray, yvArray, yArray) = Source.fromFile("data/double_spiral").getLines().map { s =>
      val x0 :: x1 :: y :: _ = s.split(" ").toList
      (x0.toDouble, x1.toDouble, y.toInt)
    }.foldLeft (Array.empty[Double], Array.empty[Double], Array.empty[Int]) {
      case ((xAcc, yvAcc, yAcc), (x0, x1, truth)) =>
        val yv = vectorize(truth, 3).data
        (xAcc :+ x0 :+ x1, yvAcc ++ yv, yAcc :+ truth)
    }

    val x = new DenseMatrix[Double](300, 2, xArray)
    val yv = new DenseMatrix[Double](300, 3, yvArray)
    val y = new DenseVector[Int](yArray)

    (x, yv, y)
  }

}
