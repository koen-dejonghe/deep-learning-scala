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

  def localUpdate(x: Tensor,
                  dx: Tensor,
                  maybeCache: Option[T]): (Tensor, Option[T])

  def update(parameters: List[Tensor],
             gradients: List[Tensor],
             maybeCaches: List[Option[T]]): (List[Tensor], List[Option[T]]) = {

    val caches =
      if (maybeCaches.isEmpty) parameters.map(_ => None) else maybeCaches

    val (newXs, newCaches) = parameters
      .zip(gradients)
      .zip(caches)
      .map {
        case ((x, dx), cache) =>
          localUpdate(x, dx, cache)
      }
      .unzip

    (newXs, newCaches)
  }

}
