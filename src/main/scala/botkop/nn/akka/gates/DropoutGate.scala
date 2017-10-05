package botkop.nn.akka.gates

import akka.actor.{Actor, ActorRef, Props}
import numsca.Tensor

import scala.language.postfixOps

class DropoutGate(next: ActorRef, p: Double) extends Actor {

  override def receive: Receive = accept()

  def accept(cache: Option[(ActorRef, Tensor)] = None): Receive = {
    case Forward(x, y) =>
      val mask = (numsca.rand(x.shape) < p) / p
      val h = x * mask
      next ! Forward(h, y)
      context become accept(Some(sender(), mask))

    case Backward(dout) if cache isDefined =>
      val (prev, mask) = cache.get
      val dx = dout * mask
      prev ! Backward(dx)

    case predict: Predict =>
      next forward predict

    case Persist =>
      next ! Persist
  }
}

object DropoutGate {
  def props(next: ActorRef, p: Double): Props =
    Props(new DropoutGate(next, p))
}
