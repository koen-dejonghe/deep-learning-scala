package botkop.nn.akka.gates

import akka.actor.{Actor, ActorRef, Props}
import numsca.Tensor

import scala.language.postfixOps

class ReluGate(next: ActorRef) extends Actor {


  // import org.nd4j.linalg.api.buffer.DataBuffer
  // import org.nd4j.linalg.api.buffer.util.DataTypeUtil
  // DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  override def receive: Receive = accept()

  def activate(z: Tensor): Tensor = numsca.maximum(z, 0.0)

  def accept(cache: Option[(ActorRef, Tensor)] = None): Receive = {
    case Forward(z, y) =>
      val a = activate(z)
      next ! Forward(a, y)
      context become accept(Some(sender(), z))

    case Backward(da) if cache isDefined =>
      val (prev, z) = cache.get
      val dz = da * (z > 0.0)
      prev ! Backward(dz)

    case Predict(z) =>
      val a = activate(z)
      next forward Predict(a)

    case Persist =>
      next ! Persist
  }
}

object ReluGate {
  def props(next: ActorRef) = Props(new ReluGate(next))
}
