package botkop.nn.cs231n

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand

import scala.io.Source
import scala.util.Random

object MinimalNet extends App {

  val N = 100 // number of points per class
  val D = 2 // dimensionality
  val K = 3 // number of classes

  val data = readData()
  val X = data._1
  val y = data._2
  val y_indices: IndexedSeq[Int] = y.data.map(_.toInt)


  val h = 100 // size of hidden layer
  val W = 0.01 * DenseMatrix.rand(D, h, Rand.gaussian)
  val b = DenseVector.zeros[Double](h)
  val W2 = 0.01 * DenseMatrix.rand(h, K, Rand.gaussian)
  val b2 = DenseVector.zeros[Double](K)

  // some hyperparameters
  val step_size = 1e-0
  val reg: Double = 1e-3 // regularization strength

  // gradient descent loop
  val num_examples = X.rows
  for (i <- 0 until 10000) {

    // evaluate class scores, [N x K]
    // hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    val dot = X * W
    dot(*, ::) :+= b
    val hidden_layer = maximum(0.0, dot)

    // scores = np.dot(hidden_layer, W2) + b2
    val scores = hidden_layer * W2
    scores(*, ::) :+= b2

    // compute the class probabilities
    // exp_scores = np.exp(scores)
    val exp_scores = exp(scores)
    // probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    val s: DenseVector[Double] = sum(exp_scores(*, ::)) //300x1
    val probs: DenseMatrix[Double] = exp_scores(::, *) / s

    // compute the loss: average cross-entropy loss and regularization
    // corect_logprobs = -np.log(probs[range(num_examples),y])
    val correct_logprobs = probs.data.grouped(probs.cols).zipWithIndex.map { case (row, rowIndex) =>
      val prob = row(y(rowIndex).toInt)
      math.log(prob) * -1.0
    }

    // data_loss = np.sum(corect_logprobs)/num_examples
    val data_loss = correct_logprobs.sum / num_examples.toDouble
    // reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    val reg_loss = 0.5 * reg * sum(W *:* W) + 0.5 * reg * sum(W2 *:* W2)
    // loss = data_loss + reg_loss
    val loss = data_loss + reg_loss
    if (i % 10 == 0) {
      println(s"iteration $i: loss $loss")
    }

    // compute the gradient on scores
    // dscores = probs
    // dscores[range(num_examples),y] -= 1
    // dscores /= num_examples

    val dscores_data: Array[Double] = probs.data.grouped(probs.cols).zipWithIndex
      .flatMap { case (row, rowIndex) =>
        val yCorrectIndex = y_indices(rowIndex)
        row(yCorrectIndex) -= 1.0
        row
      }
      .map(_ / num_examples.toDouble)
      .toArray

    val dscores = new DenseMatrix(probs.rows, probs.cols, dscores_data)


    // backpropate the gradient to the parameters
    // first backprop into parameters W2 and b2
    // dW2 = np.dot(hidden_layer.T, dscores)
    val dW2 = hidden_layer.t * dscores
    // db2 = np.sum(dscores, axis=0, keepdims=True)
    val db2 = sum(dscores, Axis._0)

    // next backprop into hidden layer
    // dhidden = np.dot(dscores, W2.T)
    val dhidden_layer = dscores * W2.t
    // backprop the ReLU non-linearity
    // dhidden[hidden_layer <= 0] = 0
    val dhidden = new DenseMatrix(
      dhidden_layer.rows,
      dhidden_layer.cols,
      dhidden_layer.data.zip(hidden_layer.data).map {
        case (dh, hl) => if (hl <= 0.0) 0.0 else dh
      })

    // finally into W,b
    // dW = np.dot(X.T, dhidden)
    // db = np.sum(dhidden, axis=0, keepdims=True)
    val dW = X.t * dhidden
    val db = sum(dhidden, Axis._0)

    // add regularization gradient contribution
    // dW2 += reg * W2
    // dW += reg * W
    dW2 :+= (W2 * reg)
    dW :+= (W * reg)

    // perform a parameter update
    // W += -step_size * dW
    // b += -step_size * db
    // W2 += -step_size * dW2
    // b2 += -step_size * db2
    W :+= (dW * - step_size)
    b :+= (db.inner * -step_size)
    W2 :+= (dW2 * - step_size)
    b2 :+= (db2.inner * -step_size)

  }

  // evaluate training set accuracy
  val dot = X * W
  dot(*, ::) :+= b
  val hidden_layer = maximum(0, dot)
  val scores = hidden_layer * W2
  scores(*, ::) :+= b2

  val predicted_class = argmax(scores, Axis._1)
  val accuracy = y_indices.zip(predicted_class.data).count { case (correct, guessed) => correct == guessed } / num_examples.toDouble
  println(s"$accuracy")

  def maximum(d: Double, m: DenseMatrix[Double]) = {
    new DenseMatrix(m.rows, m.cols, m.data.map { e => if (e < d) d else e })
  }

  def printShape(s: String, d: DenseMatrix[Double]): Unit =
    println(s"$s: ${d.rows}x${d.cols}")


  def readData() = {
    val (xArray, yArray) = Source.fromFile("data/double_spiral").getLines().map { s =>
      val x0 :: x1 :: y :: _ = s.split(" ").toList
      (x0.toDouble, x1.toDouble, y.toDouble)
    }.foldLeft(Array.empty[Double], Array.empty[Double]) {
      case ((xAcc, yAcc), (x0, x1, truth)) =>
        (xAcc :+ x0 :+ x1, yAcc :+ truth)
    }

    val x = new DenseMatrix[Double](N * K, D, xArray)
    val y = new DenseVector[Double](yArray)

    (x, y)
  }

}

/*
class Tensor(data: Array[Double], shape: List[Int]) {
  require(shape.product == data.length)
  val size: Int = data.length
  def this(shape: Int*)(data: Array[Double]) = this(data, shape.toList)


  def dot(other: Tensor): Tensor = {
    require(this.shape.length == other.shape.length)
    require(this.shape(1) == other.shape.head)

    val rs = data.grouped(shape.head)
    val cs = data.sliding(other.shape(1))

    val result: Array[Double] = rs.zip(cs).map { case (r, c) => r.zip(c).map{case (x, y) => x * y} }.sum


  }
}

object Tensor {

  def zeros(shape: Int*): Tensor = new Tensor(Array.fill(shape.product)(0.0), shape.toList)
  def ones(shape: Int*): Tensor = new Tensor(Array.fill(shape.product)(1.0), shape.toList)
  def randn(shape: Int*): Tensor = {
    val r = new Random()
    new Tensor(Array.fill(shape.product){ r.nextGaussian() }, shape.toList)
  }

}
*/
