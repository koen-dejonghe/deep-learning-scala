package botkop.nn.akka.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import numsca.Tensor

class OutputGate(costFunction: (Tensor, Tensor) => (Double, Tensor),
                 numIterations: Int,
                 listener: Option[ActorRef])
    extends Actor
    with ActorLogging {

  // import org.nd4j.linalg.api.buffer.DataBuffer
  // import org.nd4j.linalg.api.buffer.util.DataTypeUtil
  // DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  override def receive: Receive = accept()

  def accept(i: Int = 0): Receive = {

    case Forward(al, y) if i < numIterations =>
      val (cost, dal) = costFunction(al, y)
      sender() ! Backward(dal)

      if (listener.isDefined)
        listener.get ! CostLogEntry(i, cost)

      if (i % 1000 == 0) {
        log.debug(s"iteration: $i cost: $cost")
      }

      context become accept(i + 1)

    case Predict(x) =>
      /* end of the line: send the answer back */
      sender() ! x
  }
}

object OutputGate {
  def props(costFunction: (Tensor, Tensor) => (Double, Tensor),
            numIterations: Int,
            listener: Option[ActorRef] = None) =
    Props(new OutputGate(costFunction, numIterations, listener))
}

case class CostLogEntry(iteration: Int, cost: Double)
