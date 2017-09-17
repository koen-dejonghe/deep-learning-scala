package botkop.nn.akka.optimizers

import numsca.Tensor

import scala.language.postfixOps

case class Adam(shape: Array[Int],
                learningRate: Double,
                beta1: Double = 0.9,
                beta2: Double = 0.999,
                epsilon: Double = 1e-8)
    extends Optimizer {

  val shapes = List(shape, Array(shape.head, 1))

  val vs: List[Tensor] = shapes.map(shape => numsca.zeros(shape))
  val ss: List[Tensor] = shapes.map(shape => numsca.zeros(shape))

  override def update(zs: List[Tensor], dzs: List[Tensor]): List[Tensor] = {
    shapes.indices.map { i =>
      val v = vs(i)
      val s = ss(i)
      val z = zs(i)
      val dz = dzs(i)

      v *= beta1
      v += (1 - beta1) * dz

      val vCorrected = v / math.pow(1 - beta1, 2)

      s *= beta2
      s += (1 - beta2) * numsca.square(dz)

      val sCorrected = s / math.pow(1 - beta2, 2)

      z - learningRate * (vCorrected / (numsca.sqrt(sCorrected) + epsilon))
    } toList
  }

}
