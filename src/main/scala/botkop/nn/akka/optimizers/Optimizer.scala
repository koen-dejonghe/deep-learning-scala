package botkop.nn.akka.optimizers

import numsca.Tensor

trait Optimizer extends Serializable {

  /**
    * Update the parameters (weights and biases) of a Gate
    * @param parameters List of parameters where parameters(0) = weights, and parameters(1) = biases
    * @param gradients List of gradients where gradients(0) = gradients of the weights, and gradients(1) = gradients of the biases
    */
  def update(parameters: List[Tensor], gradients: List[Tensor]): Unit
}


