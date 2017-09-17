package botkop.nn.akka.gates

import akka.actor.{Actor, ActorRef, Props}
import numsca.Tensor

import scala.language.postfixOps

class ReluGate(next: ActorRef) extends Actor {

  override def receive: Receive = accept()

  def activate(z: Tensor): Tensor = numsca.maximum(z, 0.0)

  def accept(cache: Option[(ActorRef, Tensor)] = None): Receive = {
    case Forward(z) =>
      val a = activate(z)
      next ! Forward(a)
      context become accept(Some(sender(), z))

    case Backward(da) if cache isDefined =>
      val (prev, z) = cache.get
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
