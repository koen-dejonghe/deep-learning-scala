package botkop.nn.akka.gates

import akka.actor.{Actor, ActorRef, Props}
import numsca.Tensor

import scala.language.postfixOps

class InputGate(first: ActorRef) extends Actor {

  override def receive: Receive = accept()

  def accept(cache: Option[Tensor] = None): Receive = {
    case Forward(x) =>
      first ! Forward(x)
      context.become(accept(Some(x)))

    case Backward(_) if cache isDefined =>
      val x = cache.get
      sender() ! Forward(x)

    case Predict(x) =>
      first forward Predict(x)
  }
}

object InputGate {
  def props(first: ActorRef) = Props(new InputGate(first))
}
