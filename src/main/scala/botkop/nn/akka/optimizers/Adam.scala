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

  override def update(parameters: List[Tensor], gradients: List[Tensor]): List[Tensor] =
    shapes.indices.map { i =>
      val v = vs(i)
      val s = ss(i)
      val p = parameters(i)
      val dp = gradients(i)

      v *= beta1
      v += (1 - beta1) * dp

      val vCorrected = v / math.pow(1 - beta1, 2)

      s *= beta2
      s += (1 - beta2) * numsca.square(dp)

      val sCorrected = s / math.pow(1 - beta2, 2)

      p - learningRate * (vCorrected / (numsca.sqrt(sCorrected) + epsilon))
    } toList

}
