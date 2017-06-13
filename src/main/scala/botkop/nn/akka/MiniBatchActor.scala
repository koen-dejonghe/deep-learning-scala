package botkop.nn.akka

import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import akka.actor.{Actor, ActorRef, Props}

class MiniBatchActor(layer: ActorRef) extends Actor {

  var originator: ActorRef = context.parent

  override def receive: Receive = {

    case MiniBatch(batch) =>
      originator = sender
      batch.foreach {
        case (x, y) =>
          layer ! FeedForward(x, y)
      }

    case MiniBatchReady =>
      originator ! MiniBatchReady
  }

}

object MiniBatchActor {
  def props(layer: ActorRef) = Props(new MiniBatchActor(layer))
}

case class MiniBatch(batch: List[(Matrix, Matrix)])
case object MiniBatchReady
