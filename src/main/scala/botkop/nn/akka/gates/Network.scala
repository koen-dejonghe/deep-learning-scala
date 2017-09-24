package botkop.nn.akka.gates

import akka.actor.{ActorRef, ActorSystem}
import botkop.nn.akka.CostFunctions.CostFunction
import botkop.nn.akka.optimizers.Optimizer

sealed trait Layer {
  def +(other: Layer): Network = Network(List(this, other))
  def *(i: Int): Network = Network(List.fill(i)(this))
}

case object Relu extends Layer {
  def name(layer: Int) = s"relu-gate-$layer"
}

case object Sigmoid extends Layer {
  def name(layer: Int) = s"lsigmoid-gate-$layer"
}

case object Linear extends Layer {
  def name(layer: Int) = s"linear-gate-$layer"
}

case class Network(gates: List[Layer]) {
  def +(other: Network) = Network(this.gates ++ other.gates)
  def +(layer: Layer) = Network(this.gates :+ layer)
  def *(i: Int): Network = Network(List.tabulate(i)(_ => gates).flatten)
}

object Network {

  def initialize(layout: Network,
                 dimensions: Array[Int],
                 miniBatchSize: Int,
                 regularization: Double,
                 optimizer: => Optimizer,
                 costFunction: CostFunction,
                 numIterations: Int = Int.MaxValue): (ActorRef, ActorRef) = {

    val system: ActorSystem = ActorSystem()

    val numLinearGates = layout.gates.count(g => g == Linear)
    require(numLinearGates == dimensions.length - 1)

    val output = system.actorOf(OutputGate.props(costFunction, numIterations))

    val (first, _) =
      layout.gates.reverse.foldLeft(output, dimensions.length) {
        case ((next, i), gate) =>
          gate match {
            case Relu =>
              val reluGate = system.actorOf(ReluGate.props(next), Relu.name(i))
              (reluGate, i)

            case Sigmoid =>
              val sigmoidGate =
                system.actorOf(SigmoidGate.props(next), Sigmoid.name(i))
              (sigmoidGate, i)

            case Linear =>
              val shape = dimensions.slice(i - 2, i).reverse
              val linearGate = system.actorOf(
                LinearGate.props(shape, next, regularization, optimizer),
                Linear.name(i))
              (linearGate, i - 1)
          }
      }

    val input =
      system.actorOf(InputGate.props(first, miniBatchSize = miniBatchSize))

    (input, output)
  }

}
