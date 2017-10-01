package botkop.nn.akka.optimizers

import numsca.Tensor

case class Momentum(learningRate: Double, beta: Double = 0.9)
    extends Optimizer {

  var vs = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit = {

    if (vs.isEmpty) {
      vs = parameters.map(numsca.zerosLike)
    }

    parameters.zip(gradients).zipWithIndex.foreach {
      case ((z, dz), i) =>
        vs(i) *= beta
        vs(i) += (1 - beta) * dz
        z -= learningRate * vs(i)
    }

  }
}
