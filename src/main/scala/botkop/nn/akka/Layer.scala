package botkop.nn.akka

import akka.actor.{Actor, ActorRef, Props}
import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

case class Shape(n: Int, m: Int)

class Layer(shape: Shape, hp: HyperParameters) extends Actor {

  val weights: Matrix = randn(shape.m, shape.n) / Math.sqrt(shape.n)
  val biases: Matrix = randn(shape.m, 1)
  val incomingActivations: Matrix = zeros(shape.n, 1)
  val activations: Matrix = zeros(shape.m, 1)
  val nablaWeights: Matrix = zeros(shape.m, shape.n)
  val nablaBiases: Matrix = zeros(shape.m, 1)

  var counter = 0

  override def receive: Receive = {
    case Wiring(fwd, bwd) =>
      context.become(wired(fwd, bwd))
  }

  def wired(fwd: Option[ActorRef], bwd: Option[ActorRef]): Receive = {

    case FeedForward(x, y) =>
      incomingActivations.assign(x)
      activations.assign(sigmoid((weights dot x) + biases))
      fwd match {
        case Some(actor) =>
          actor forward FeedForward(activations, y)
        case None =>
          // output layer
          val delta = activations - y // cross entropy cost
          val nb = delta
          val nw = delta dot incomingActivations.transpose()
          nablaBiases += nb
          nablaWeights += nw

          bwd.get forward DeltaBackward(weights.transpose() dot nb)
      }

    case DeltaBackward(d) =>
      counter += 1
//      println(s"bwd $counter ")
      val sp = derivative(activations)

      /*
      bwd match {
        case Some(actor) =>
          //actor ! DeltaBackward(weights.transpose() dot delta)
          println("AAAAAAAAAAAAAAAAAAAAA")
        case None => // input layer
      }
      */

      val delta = d * sp

      val nb = delta
      val nw = delta dot incomingActivations.transpose()

      nablaBiases += nb
      nablaWeights += nw

      if (counter >= hp.miniBatchSize) {
        counter = 0
        self ! UpdateWeightsAndBiases
        sender ! MiniBatchReady
      }

    case UpdateWeightsAndBiases =>
      fwd match {
        case Some(fwdLayer) =>
          fwdLayer ! UpdateWeightsAndBiases
        case None =>
      }
      updateWeightsAndBiases()

    case Guess(x) =>
      val z = (weights dot x) + biases
      val a = sigmoid(z)

//      println("a = " + a)

      fwd match {
        case Some(fwdLayer) =>
          fwdLayer forward Guess(a)
        case None =>
          sender ! a
      }

  }

  def updateWeightsAndBiases(): Unit = {
    biases -= nablaBiases * hp.lm
    weights *= hp.lln
    weights -= nablaWeights * hp.lm

    nablaWeights.assign(zeros(shape.m, shape.n))
    nablaBiases.assign(zeros(shape.m, 1))
  }

  def derivative(z: Matrix): Matrix = z * (-z + 1.0)
}

object Layer {
  def props(shape: Shape, hp: HyperParameters): Props =
    Props(new Layer(shape, hp))
}

case class FeedForward(x: Matrix, y: Matrix)
case class DeltaBackward(delta: Matrix)
case object UpdateWeightsAndBiases

case class Guess(x: Matrix)

case class Wiring(fwd: Option[ActorRef], bwd: Option[ActorRef])
