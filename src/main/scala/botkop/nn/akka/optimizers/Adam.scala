package botkop.nn.akka.optimizers

import numsca.Tensor

import scala.language.postfixOps


case class Adam(learningRate: Double,
                beta1: Double = 0.9,
                beta2: Double = 0.999,
                epsilon: Double = 1e-8)
    extends Optimizer {

  var t = 1

  var vs = List.empty[Tensor]
  var ss = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): List[Tensor] = {

    // first time through
    // create the cache
    if (t == 1) {
      vs = parameters.map(numsca.zerosLike)
      ss = parameters.map(numsca.zerosLike)
    }

    t = t + 1

    parameters.indices.map { i =>
      val p = parameters(i)
      val dp = gradients(i)

      vs(i) *= beta1
      vs(i) += (1 - beta1) * dp

      val vCorrected = vs(i) / (1 - math.pow(beta1, t))

      ss(i) *= beta2
      ss(i) += (1 - beta2) * numsca.square(dp)

      val sCorrected = ss(i) / (1 - math.pow(beta2, t))

      p + (-learningRate * vCorrected / (numsca.sqrt(sCorrected) + epsilon))

    } toList
  }

}

