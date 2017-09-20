package botkop.nn.akka.gates

import akka.actor.{Actor, ActorLogging, Props}
import numsca.Tensor

class OutputGate(costFunction: (Tensor, Tensor) => (Double, Tensor),
                 numIterations: Int)
    extends Actor with ActorLogging {

  override def receive: Receive = accept()

  def accept(i: Int = 0): Receive = {

    case Forward(al, y) if i < numIterations =>
      val (cost, dal) = costFunction(al, y)
      sender() ! Backward(dal)

      if (i % 100 == 0) {
        log.debug(s"cost at iteration $i: $cost")
      }
      context become accept(i + 1)

    case Predict(x) =>
      /* end of the line: send the answer back */
      sender() ! x
  }

}

object OutputGate {
  def props(costFunction: (Tensor, Tensor) => (Double, Tensor),
            numIterations: Int) =
    Props(new OutputGate(costFunction, numIterations))
}
