package botkop.nn.akka.optimizers

import numsca.Tensor

trait Optimizer {
  def update(zs: List[Tensor], dzs: List[Tensor]): List[Tensor]
}
