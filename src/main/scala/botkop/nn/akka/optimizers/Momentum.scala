package botkop.nn.akka.optimizers

import numsca.Tensor

case class Momentum(shape: Array[Int], beta: Double, learningRate: Double)
    extends Optimizer {

  val shapes = List(shape, Array(shape.head, 1))

  val vs: List[Tensor] = shapes.map(shape => numsca.zeros(shape))
  val ss: List[Tensor] = shapes.map(shape => numsca.zeros(shape))

  override def update(zs: List[Tensor], dzs: List[Tensor]): List[Tensor] = {
    zs.zip(dzs).zip(vs).map {
      case ((z, dz), v) =>
        v *= beta
        v += (1 - beta) * dz
        z - learningRate * v
    }
  }
}
