package gate

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import numsca.Tensor

import scala.io.Source

object ActorNetwork extends App {

  val system = ActorSystem()

  val xTrain =
    readData("data/coursera/catvsnoncat/train_x.csv", Array(12288, 209))
  val yTrain =
    readData("data/coursera/catvsnoncat/train_y.csv", Array(1, 209))

  val dimensions = Array(12288, 20, 7, 5, 1)
  val learningRate = 0.0075

  val (input, output) = initialize(dimensions, yTrain, learningRate)

  input ! Forward(xTrain)

  def crossEntropyCost(yHat: Tensor, y: Tensor): Double = {
    val m = y.shape(1)
    val cost = (-y.dot(numsca.log(yHat).T) -
      (1 - y).dot(numsca.log(1 - yHat).T)) / m
    cost.squeeze()
  }

  def initialize(dimensions: Array[Int],
                 y: Tensor,
                 learningRate: Double): (ActorRef, ActorRef) = {

    val output =
      system.actorOf(OutputGate.props(y, crossEntropyCost, 2500), "output")

    val nl = system.actorOf(SigmoidGate.props(output), "nl")

    val first = system.actorOf(
      LinearGate.props(Array(1, 12288), nl, learningRate),
      "input")

    val input = system.actorOf(InputGate.props(first))

    /*
    val (_, input) = dimensions.reverse.sliding(2).foldLeft(true, output) {
      case ((isLast, next), shape) =>
        val nonLinearity =
          if (isLast) system.actorOf(SigmoidGate.props(next))
          else system.actorOf(ReluGate.props(next))

        (false,
         system.actorOf(LinearGate.props(shape, nonLinearity, learningRate)))
    }
     */

    (input, output)

  }

  def readData(fileName: String, shape: Array[Int]): Tensor = {
    val data = Source
      .fromFile(fileName)
      .getLines()
      .map(_.split(",").map(_.toDouble))
      .flatten
      .toArray
    Tensor(data).reshape(shape)
  }

}

class LinearGate(shape: Array[Int], next: ActorRef, learningRate: Double)
    extends Actor {

  override def receive: Receive = {
    val w = numsca.randn(shape) * math.sqrt(2.0 / shape(1))
    val b = numsca.zeros(shape.head, 1)
    fwd(w, b)
  }

  def fwd(w: Tensor, b: Tensor): Receive = {
    case Forward(x) =>
      val z = w.dot(x) + b
      next ! Forward(z)
      context become bwd(sender(), x, w, b)
  }

  def bwd(prev: ActorRef, a: Tensor, w: Tensor, b: Tensor): Receive = {
    case Backward(dz) =>
      val da = w.transpose.dot(dz)
      prev ! Backward(da)

      val m = a.shape(1)
      val dw = dz.dot(a.T) / m
      val db = numsca.sum(dz, axis = 1) / m
      context become fwd(w - dw * learningRate, b - db * learningRate)
  }
}

object LinearGate {
  def props(shape: Array[Int], next: ActorRef, learningRate: Double) =
    Props(new LinearGate(shape, next, learningRate))
}

class ReluGate(next: ActorRef) extends Actor {

  override def receive: Receive = fwd

  def fwd: Receive = {
    case Forward(z) =>
      println("bbb")
      val a = numsca.maximum(z, 0.0)
      next ! Forward(a)
      context become bwd(sender(), z)
  }

  def bwd(prev: ActorRef, z: Tensor): Receive = {
    case Backward(da) =>
      val dz = da * (z > 0.0)
      prev ! Backward(dz)
      context become fwd
  }
}

object ReluGate {
  def props(next: ActorRef) = Props(new ReluGate(next))
}

class SigmoidGate(next: ActorRef) extends Actor {
  override def receive: Receive = fwd

  def fwd: Receive = {
    case Forward(z) =>
      val s = numsca.sigmoid(z)
      next ! Forward(s)
      context become bwd(sender(), s)
  }

  def bwd(prev: ActorRef, s: Tensor): Receive = {
    case Backward(da: Tensor) =>
      val dz = da * s * (1 - s)
      prev ! Backward(dz)
      context become fwd
  }
}

object SigmoidGate {
  def props(next: ActorRef) = Props(new SigmoidGate(next))
}

class OutputGate(y: Tensor,
                 costFunction: (Tensor, Tensor) => Double,
                 numIterations: Int)
    extends Actor {

  override def receive: Receive = iteration(0)

  def iteration(i: Int): Receive = {
    case Forward(al) =>
      if (i % 100 == 0) {
        val cost = costFunction(al, y)
        println(s"cost at iteration $i: $cost")
      }

      if (i < numIterations) {
        val ys = y.reshape(al.shape)
        val dal = -(ys / al - (1 - ys) / (1 - al))
        sender() ! Backward(dal)
        context become iteration(i + 1)
      } else {
        context.system.terminate()
      }
  }

}

object OutputGate {
  def props(y: Tensor,
            costFunction: (Tensor, Tensor) => Double,
            numIterations: Int) =
    Props(new OutputGate(y, costFunction, numIterations))
}

class InputGate(first: ActorRef) extends Actor {
  override def receive: Receive = {
    case Forward(x) =>
      first ! Forward(x)
      context become bwd(x)
  }

  def bwd(x: Tensor): Receive = {
    case Backward(dx) =>
      // do nothing with dx
      sender ! Forward(x)
  }
}

object InputGate {
  def props(first: ActorRef) = Props(new InputGate(first))
}

case class Forward(z: Tensor)
case class Backward(dz: Tensor)
