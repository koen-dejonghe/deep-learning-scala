package botkop.nn

import breeze.linalg.{DenseMatrix => Matrix}
import breeze.numerics.{abs, exp}
import breeze.stats.distributions.Rand

import breeze.linalg._
import breeze.stats._
import breeze.numerics._

/*
https://iamtrask.github.io/2015/07/12/basic-python-network/
 */
object Simple extends App {

  val X = Matrix(
    (0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)
  )

  val y = Matrix(
    (0.0, 1.0, 1.0, 0.0)
  ).t

  val W0 = Matrix.rand(3, 4, Rand.gaussian)
  val W1 = Matrix.rand(4, 1, Rand.gaussian)

  for (j <- 1 to 60000) {
    // forward pass
    val l0 = X
    val l1 = sigmoid(l0 * W0)
    val l2 = sigmoid(l1 * W1)

    relu(l0*W0)

    // how much did we miss the target value?
    val l2_error = y - l2

    if (j % 10000 == 0) {
      println(mean(abs(l2_error)))
    }

    // output layer
    // in what direction is the target value?
    val derivative_2 = sigmoidPrime(l2) // local derivative
    val l2_delta = l2_error *:* derivative_2

    // hidden layer
    val l1_error = l2_delta * W1.t
    val derivative_1 = sigmoidPrime(l1) // local derivative
    val l1_delta = l1_error *:* derivative_1

    W1 :+= l1.t * l2_delta
    W0 :+= X.t * l1_delta


  }

  // activation function
  private def sigmoid(x: Matrix[Double]) = 1.0 / (exp(-x) + 1.0)

  // derivative of the activation function
  private def sigmoidPrime(x: Matrix[Double]) = x *:* (1.0 - x)

}

