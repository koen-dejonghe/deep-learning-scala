package coursera.deeplearning.course2.week2

import numsca._

import scala.language.postfixOps
import scala.util.Random

object OptimizationModel {

  type TensorMap = Map[String, Tensor]

  /**
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters

    */
  def updateParametersWithGd(parameters: TensorMap,
                             grads: TensorMap,
                             learningRate: Double): TensorMap = {

    val numLayers = parameters.size / 2 // number of layers in the neural networks

    for (l <- 0 until numLayers) {
      parameters(s"W$l") -= learningRate * grads(s"dW$l")
      parameters(s"b$l") -= learningRate * grads(s"db$l")
    }
    parameters
  }

  /**
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    */
  def randomMiniBatches(x: Tensor,
                        y: Tensor,
                        miniBatchSize: Int = 64,
                        seed: Int = 0): (List[Tensor], List[Tensor]) = {

    val m = x.shape(1)

    Random.setSeed(seed)
    val permutation = Random.shuffle((0 :> m).toList)

    val shuffledX = x(:>, permutation)
    val shuffledY = y(:>, permutation)

    (0 :> m)
      .sliding(miniBatchSize, miniBatchSize)
      .foldLeft((List.empty[Tensor], List.empty[Tensor])) {
        case ((xs, ys), slice) =>
          (xs :+ shuffledX(:>, slice), ys :+ shuffledY(:>, slice))
      }
  }

  /**
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    */
  def initializeVelocity(parameters: TensorMap): TensorMap = {

    val numLayers = parameters.size / 2

    (1 to numLayers).foldLeft(Map.empty[String, Tensor]) {
      case (m, l) =>
        m ++ Seq(
          s"dW$l" -> zerosLike(parameters(s"W$l")),
          s"db$l" -> zerosLike(parameters(s"b$l"))
        )
    }
  }

  /**
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    */
  def updateParametersWithMomentum(
      parameters: TensorMap,
      grads: TensorMap,
      v: TensorMap,
      beta: Double,
      learningRate: Double): (TensorMap, TensorMap) = {

    val numLayers = parameters.size / 2

    // todo use fold left or so
    for (l <- 1 to numLayers) {
      // compute velocities
      v(s"dW$l") *= beta
      v(s"dW$l") += (1 - beta) * grads(s"dW$l")
      v(s"db$l") *= beta
      v(s"db$l") += (1 - beta) * grads(s"db$l")
      // update parameters
      parameters(s"W$l") -= learningRate * v(s"dW$l")
      parameters(s"b$l") -= learningRate * v(s"db$l")
    }

    (parameters, v)
  }

  /**
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
    */
  def initializeAdam(parameters: TensorMap): (TensorMap, TensorMap) = {

    val numLayers = parameters.size / 2

    (1 to numLayers).foldLeft(Map.empty[String, Tensor],
                              Map.empty[String, Tensor]) {
      case ((v, s), l) =>
        (v ++ Seq(
           s"dW$l" -> zerosLike(parameters(s"W$l")),
           s"db$l" -> zerosLike(parameters(s"b$l"))
         ),
         s ++ Seq(
           s"dW$l" -> zerosLike(parameters(s"W$l")),
           s"db$l" -> zerosLike(parameters(s"b$l"))
         ))
    }
  }

  /**
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    */
  def updateParametersWithAdam(
      parameters: TensorMap,
      grads: TensorMap,
      v: TensorMap,
      s: TensorMap,
      learningRate: Double = 0.01,
      beta1: Double = 0.9,
      beta2: Double = 0.999,
      epsilon: Double = 1e-8): (TensorMap, TensorMap, TensorMap) = {

    val numLayers = parameters.size / 2

    for (l <- 1 to numLayers) {
      // Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
      v(s"dW$l") *= beta1
      v(s"dW$l") += (1 - beta1) * grads(s"dW$l")
      v(s"db$l") *= beta1
      v(s"db$l") += (1 - beta1) * grads(s"db$l")

      // Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
      val vCorrectedW = v(s"dW$l") / math.pow(1 - beta1, 2)
      val vCorrectedB = v(s"db$l") / math.pow(1 - beta1, 2)

      // Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
      s(s"dW$l") *= beta2
      s(s"dW$l") += (1 - beta2) * square(grads(s"dW$l"))
      s(s"db$l") *= beta2
      s(s"db$l") += (1 - beta2) * square(grads(s"db$l"))

      // Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
      val sCorrectedW = v(s"dW$l") / math.pow(1 - beta2, 2)
      val sCorrectedB = v(s"db$l") / math.pow(1 - beta2, 2)

      // Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
      parameters(s"W$l") -= learningRate * (vCorrectedW / (sqrt(sCorrectedW) + epsilon))
      parameters(s"b$l") -= learningRate * (vCorrectedB / (sqrt(sCorrectedB) + epsilon))
    }

    (parameters, v, s)
  }

}
