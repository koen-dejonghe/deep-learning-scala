package botkop.nn.akka.optimizers

import numsca.Tensor

import scala.language.postfixOps

case class Nesterov(var learningRate: Double,
                    learningRateDecay: Double = 1.0 - 1e-8,
                    beta: Double = 0.9)
    extends Optimizer {

  var vs = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): List[Tensor] = {

    if (vs.isEmpty) {
      vs = parameters.map(numsca.zerosLike)
    }

    learningRate *= learningRateDecay

    parameters.indices.map { i =>
      update(parameters(i), gradients(i), vs(i))
    } toList
  }

  def update(x: Tensor, dx: Tensor, v: Tensor): Tensor = {
    val vPrev = v
    v *= beta
    v -= learningRate * dx
    x + (-beta * vPrev) + (1 + beta) * v
  }
}
