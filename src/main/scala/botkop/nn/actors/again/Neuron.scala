package botkop.nn.actors.again

import akka.actor.{Actor, ActorRef}
import breeze.linalg.DenseVector
import breeze.numerics.sigmoid
import breeze.stats.distributions.Rand

import scala.util.Random

class Neuron(size: Int) extends Actor {

  override def receive: Receive =
    run(DenseVector.rand(size, Rand.gaussian), Random.nextGaussian(), 0.0)

  def run(weights: DenseVector[Double], bias: Double, activation: Double): Receive = {
    case Activate(x, nextLayer) =>
      val activation = sigmoid(x * weights + bias).apply(0)
      nextLayer ! Activation(activation)
      context.become(run(weights, bias, activation))

    case Update(nbl, y, prevLayer) =>
      val sp = activation * (1.0 - activation)
      val db = (weights * nbl) * activation





  }
}

case class Activate(x: DenseVector[Double], target: ActorRef)
case class Activation(x: Double)
case class Update(nbl: Double, y: DenseVector[Double], target: ActorRef)
