package botkop.nn.akka.optimizers

import numsca.Tensor

case class GradientDescent(learningRate: Double) extends Optimizer {

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit =
    parameters.zip(gradients).foreach {
      case (z, dz) =>
        z -= dz * learningRate
    }
}
