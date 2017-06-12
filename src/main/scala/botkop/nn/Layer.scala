package botkop.nn

import akka.actor.{Actor, ActorRef, Props}
import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

case class Shape(n: Int, m: Int)

class Layer(shape: Shape, hp: HyperParameters) extends Actor {

  val weights: Matrix = randn(shape.m, shape.n) / Math.sqrt(shape.n)
  val biases: Matrix = randn(shape.m, 1)
  val activations: Matrix = zeros(shape.m, 1)
  val nablaWeights: Matrix = zeros(shape.m, shape.n)
  val nablaBiases: Matrix = zeros(shape.m, 1)

  override def receive: Receive = {
    case Wiring(fwd, bwd) =>
      context.become(wired(fwd, bwd))
  }

  def wired(fwd: Option[ActorRef], bwd: Option[ActorRef]): Receive = {

    case FeedForward(x, y) =>
      val z = (weights dot x) + biases
      val a = sigmoid(z)
      activations.assign(a)
      fwd match {
        case Some(actor) =>
          actor ! FeedForward(a, y)
        case None =>
          // output layer
          val delta = a - y // cross entropy cost
          bwd.get ! DeltaBackward(a, delta)
      }

    case DeltaBackward(a, d) =>
      val sp = derivative(a)
      val delta = (weights.transpose() dot d) * sp

      val nb = delta
      val nw = delta dot activations.transpose()

      nablaBiases += nb
      nablaWeights += nw

      bwd match {
        case Some(actor) =>
          actor ! DeltaBackward(activations, delta)
        case None => // input layer
      }

    case UpdateWeightsAndBiases =>
      fwd match {
        case Some(actor) =>
          actor ! UpdateWeightsAndBiases
        case None => // done
      }

      biases -= (nablaBiases * hp.lm)
      weights *= hp.lln
      weights -= nablaWeights * hp.lm


    case Guess(x, collector) =>
      val z = (weights dot x) + biases
      val a = sigmoid(z)
      fwd match {
        case Some(fwdLayer) =>
          Guess(a, collector)
        case None =>
          collector ! a
      }

  }

  def derivative(z: Matrix): Matrix = z * (-z + 1.0)
}

object Layer {
  def props(shape: Shape, hp: HyperParameters): Props =
    Props(new Layer(shape, hp))
}

case class FeedForward(x: Matrix, y: Matrix)
case class DeltaBackward(a: Matrix, delta: Matrix)
case object UpdateWeightsAndBiases

case class Guess(x: Matrix, collector: ActorRef)

case class Wiring(fwd: Option[ActorRef], bwd: Option[ActorRef])
