package botkop.nn.akka.gates

import akka.actor.{Actor, Props}
import numsca.Tensor

class OutputGate(y: Tensor,
                 costFunction: (Tensor, Tensor) => (Double, Tensor),
                 numIterations: Int)
    extends Actor {

  override def receive: Receive = accept()

  def accept(i: Int = 0): Receive = {

    case Forward(al) if i < numIterations =>
      val (cost, dal) = costFunction(al, y)
      sender() ! Backward(dal)

      if (i % 100 == 0) {
        println(s"cost at iteration $i: $cost")
      }
      context become accept(i + 1)

    case Predict(x) =>
      sender() ! x

  }

}

object OutputGate {
  def props(y: Tensor,
            costFunction: (Tensor, Tensor) => (Double, Tensor),
            numIterations: Int) =
    Props(new OutputGate(y, costFunction, numIterations))
}
