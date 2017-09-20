package botkop.nn.akka.gates

import akka.actor.{Actor, ActorRef, Props}
import numsca._

import scala.language.postfixOps
import scala.util.Random

class InputGate(first: ActorRef, miniBatchSize: Int, seed: Long) extends Actor {

  import org.nd4j.linalg.api.buffer.DataBuffer
  import org.nd4j.linalg.api.buffer.util.DataTypeUtil

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  Random.setSeed(seed)

  override def receive: Receive = accept()

  def accept(cache: Option[(Tensor, Tensor)] = None): Receive = {
    case Forward(x, y) =>
      // for (_ <- 1 to 4)
        first ! nextBatch(x, y)
      context.become(accept(Some(x, y)))

    case Backward(_) if cache isDefined =>
      val (x, y) = cache.get

      sender() ! nextBatch(x, y)

    case Predict(x) =>
      first forward Predict(x)
  }

  def nextBatch(x: Tensor, y: Tensor): Forward = {
    val m = x.shape(1)
    if (miniBatchSize >= m) {
      Forward(x, y)
    } else {
      val samples = Random.shuffle((0 until m).toList).take(miniBatchSize)
      // this is very slow
      // Forward(x(:>, samples), y(:>, samples))

      val xb = new Tensor(x.array.getColumns(samples: _*))
      val yb = new Tensor(y.array.getColumns(samples: _*))

      Forward(xb, yb)
    }
  }

}

object InputGate {
  def props(first: ActorRef, miniBatchSize: Int = 64, seed: Long = 231) =
    Props(new InputGate(first, miniBatchSize, seed))
}
