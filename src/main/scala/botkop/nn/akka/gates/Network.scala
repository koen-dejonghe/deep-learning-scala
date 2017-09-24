package botkop.nn.akka.gates

import akka.actor.{ActorRef, ActorSystem}
import botkop.nn.akka.CostFunctions._
import botkop.nn.akka.optimizers.{GradientDescent, Optimizer}

sealed trait Gate {
  def +(other: Gate): Network = Network(List(this, other))
  def *(i: Int): Network = Network(List.fill(i)(this))
}

case object Relu extends Gate {
  def name(layer: Int) = s"relu-gate-$layer"
}

case object Sigmoid extends Gate {
  def name(layer: Int) = s"sigmoid-gate-$layer"
}

case object Linear extends Gate {
  def name(layer: Int) = s"linear-gate-$layer"
}

case class Network(gates: List[Gate] = List.empty,
                   dimensions: Array[Int] = Array.empty,
                   miniBatchSize: Int = 16,
                   costFunction: CostFunction = crossEntropyCost,
                   optimizer: () => Optimizer = () =>
                     GradientDescent(learningRate = 0.01),
                   regularization: Double = 0.0,
                   maxIterations: Int = Int.MaxValue) {

  def +(other: Network) = Network(this.gates ++ other.gates)
  def +(layer: Gate) = Network(this.gates :+ layer)
  def *(i: Int): Network = Network(List.tabulate(i)(_ => gates).flatten)

  def withGates(gs: Gate*): Network = copy(gates = gs.toList)
  def withDimensions(dims: Int*): Network = copy(dimensions = dims.toArray)
  def withMiniBatchSize(bs: Int): Network = copy(miniBatchSize = bs)
  def withCostFunction(cf: CostFunction): Network = copy(costFunction = cf)
  def withOptimizer(o: () => Optimizer): Network = copy(optimizer = o)
  def withRegularization(reg: Double): Network = copy(regularization = reg)
  def withMaxIterations(max: Int): Network = copy(maxIterations = max)

  def init(): (ActorRef, ActorRef) = {

    require(gates.nonEmpty)
    require(dimensions.nonEmpty)
    val numLinearGates = gates.count(g => g == Linear)
    require(numLinearGates == dimensions.length - 1)

    val system: ActorSystem = ActorSystem()

    val output = system.actorOf(OutputGate.props(costFunction, maxIterations))

    val (first, _) =
      gates.reverse.foldLeft(output, dimensions.length) {
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
                LinearGate.props(shape, next, regularization, optimizer()),
                Linear.name(i))
              (linearGate, i - 1)
          }
      }

    val input =
      system.actorOf(InputGate.props(first, miniBatchSize = miniBatchSize))

    (input, output)
  }
}
