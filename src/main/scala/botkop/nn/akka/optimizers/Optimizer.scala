package botkop.nn.akka.optimizers

import numsca.Tensor

trait Optimizer {

  /**
    * Update the parameters (weights and biases) of a Gate
    * @param parameters List of parameters where parameters(0) = weights, and parameters(1) = biases
    * @param gradients List of gradients where gradients(0) = gradients of the weights, and gradients(1) = gradients of the biases
    * @return List of updated weights and biases
    */
  def update(parameters: List[Tensor], gradients: List[Tensor]): (List[Tensor])
}

trait OptimizerCache

trait Optimizer2[T >: OptimizerCache] {

  def update(
      parameters: List[Tensor],
      gradients: List[Tensor],
      maybeCaches: Option[List[T]]): (List[Tensor], Option[List[T]])

}
