package botkop.nn.akka.gates

import akka.actor.{Actor, ActorRef, Props}
import botkop.nn.akka.optimizers.Optimizer
import numsca.Tensor

import scala.language.postfixOps

class LinearGate(shape: Array[Int],
                 next: ActorRef,
                 optimizer: Optimizer,
                 // learningRate: Double,
                 seed: Option[Long] = Some(231))
    extends Actor {

  if (seed isDefined) numsca.rand.setSeed(seed.get)

  override def receive: Receive = {
    val w = numsca.randn(shape) * math.sqrt(2.0 / shape(1))
    val b = numsca.zeros(shape.head, 1)
    accept(w, b)
  }

  def activate(x: Tensor, w: Tensor, b: Tensor): Tensor = w.dot(x) + b

  def accept(w: Tensor,
             b: Tensor,
             cache: Option[(ActorRef, Tensor)] = None): Receive = {

    case Forward(x) =>
      val z = activate(x, w, b)
      next ! Forward(z)
      context become accept(w, b, Some(sender(), x))

    case Predict(x) =>
      val z = activate(x, w, b)
      next forward Predict(z)

    case Backward(dz) if cache isDefined =>
      val (prev, a) = cache.get

      val da = w.transpose.dot(dz)
      prev ! Backward(da)

      val m = a.shape(1)
      val dw = dz.dot(a.T) / m
      val db = numsca.sum(dz, axis = 1) / m

      val List(updatedW, updatedB) = optimizer.update(List(w, b), List(dw, db))
      context become accept(updatedW, updatedB, cache)

      // context become accept(w - dw * learningRate, b - db * learningRate, cache)
  }

}

object LinearGate {
  def props(shape: Array[Int], next: ActorRef, optimizer: Optimizer) =
    Props(new LinearGate(shape, next, optimizer))
  // def props(shape: Array[Int], next: ActorRef, learningRate: Double) =
    // Props(new LinearGate(shape, next, learningRate))
}