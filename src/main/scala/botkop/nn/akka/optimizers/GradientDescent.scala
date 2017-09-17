package botkop.nn.akka.optimizers

import numsca.Tensor

case class GradientDescent(learningRate: Double) extends Optimizer {

  override def update(zs: List[Tensor], dzs: List[Tensor]): List[Tensor] =
    zs.zip(dzs).map {
      case (z, dz) =>
        z - dz * learningRate
    }
}


