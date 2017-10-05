package botkop.nn.akka.gates

import akka.actor.{Actor, ActorRef, Props}
import numsca.Tensor

import scala.language.postfixOps

class SigmoidGate(next: ActorRef) extends Actor {

  // import org.nd4j.linalg.api.buffer.DataBuffer
  // import org.nd4j.linalg.api.buffer.util.DataTypeUtil
  // DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)


  override def receive: Receive = accept()

  def activate(z: Tensor): Tensor = numsca.sigmoid(z)

  def accept(cache: Option[(ActorRef, Tensor)] = None): Receive = {

    case Forward(z, y) =>
      val a = activate(z)
      next ! Forward(a, y)
      context become accept(Some(sender(), a)) // !!! note passing the sigmoid in the cache

    case Backward(da) if cache isDefined =>
      val (prev, s) = cache.get
      val dz = da * s * (1 - s)
      prev ! Backward(dz)

    case Predict(x) =>
      val a = activate(x)
      next forward Predict(a)
  }
}

object SigmoidGate {
  def props(next: ActorRef) = Props(new SigmoidGate(next))
}
