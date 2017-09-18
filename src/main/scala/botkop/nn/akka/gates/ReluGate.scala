package botkop.nn.akka.gates

import akka.actor.{Actor, ActorRef, Props}
import numsca.Tensor

import scala.language.postfixOps

class ReluGate(next: ActorRef) extends Actor {

  override def receive: Receive = accept()

  def activate(z: Tensor): Tensor = numsca.maximum(z, 0.0)

  def accept(cache: Option[(ActorRef, Tensor, Tensor)] = None): Receive = {
    case Forward(z, y) =>
      val a = activate(z)
      next ! Forward(a, y)
      context become accept(Some(sender(), z, y))

    case Backward(da) if cache isDefined =>
      val (prev, z, _) = cache.get
      val dz = da * (z > 0.0)
      prev ! Backward(dz)

    case Predict(z) =>
      val a = activate(z)
      next forward Predict(a)
  }

}

object ReluGate {
  def props(next: ActorRef) = Props(new ReluGate(next))
}
