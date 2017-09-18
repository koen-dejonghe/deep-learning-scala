package botkop.nn.akka.optimizers

import numsca.Tensor

case class Momentum(shape: Array[Int], learningRate: Double, beta: Double = 0.9)
    extends Optimizer {

  val shapes = List(shape, Array(shape.head, 1))

  val vs: List[Tensor] = shapes.map(shape => numsca.zeros(shape))

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): List[Tensor] = {
    parameters.zip(gradients).zip(vs).map {
      case ((z, dz), v) =>
        v *= beta
        v += (1 - beta) * dz
        z - learningRate * v
    }
  }
}
