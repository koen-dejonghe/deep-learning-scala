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

    for (l <- 1 to numLayers) {
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
    val permutation: List[Int] = Random.shuffle((0 :> m).toList)

    val shuffledX = x(:>, permutation)
    val shuffledY = y(:>, permutation)

    (0 until m)
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
      val sCorrectedW = s(s"dW$l") / math.pow(1 - beta2, 2)
      val sCorrectedB = s(s"db$l") / math.pow(1 - beta2, 2)

      // Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
      parameters(s"W$l") -= learningRate * (vCorrectedW / (sqrt(sCorrectedW) + epsilon))
      parameters(s"b$l") -= learningRate * (vCorrectedB / (sqrt(sCorrectedB) + epsilon))
    }

    (parameters, v, s)
  }

  /**
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    */
  def model(x: Tensor,
            y: Tensor,
            layersDims: Array[Int],
            optimizer: String,
            learningRate: Double = 0.0007,
            miniBatchSize: Int = 64,
            beta: Double = 0.9,
            beta1: Double = 0.9,
            beta2: Double = 0.999,
            epsilon: Double = 1e-8,
            numEpochs: Int = 10000,
            seed: Int = 10,
            printCost: Boolean = true): Map[String, Tensor] = {

    // disclaimer: this is a translation of a python program,
    // and is by no means an example of functional programming

    import OptUtils._

    // Initialize parameters
    var parameters = initializeParameters(layersDims)

    // Initialize the optimizer
    var (v, s) = optimizer match {
      case "momentum" =>
        (initializeVelocity(parameters), Map.empty[String, Tensor])
      case "adam" =>
        initializeAdam(parameters)
      case _ => // no initialization required for gradient descent
        (Map.empty[String, Tensor], Map.empty[String, Tensor])
    }

    // Optimization loop
    for (i <- 1 to numEpochs) {

      // Define the random minibatches.
      // We increment the seed to reshuffle differently the dataset after each epoch
      val miniBatches = {
        val (miniBatchesX, miniBatchesY) =
          randomMiniBatches(x, y, miniBatchSize, seed + i)
        miniBatchesX.zip(miniBatchesY)
      }

      var cost = 0.0

      miniBatches.foreach {
        case (miniBatchX, miniBatchY) =>
          // Forward propagation
          val (a3, caches) = forwardPropagation(miniBatchX, parameters)

          // Compute cost
          // cost = computeCost(a3, miniBatchY)

          // Backward propagation
          val grads = backwardPropagation(miniBatchX, miniBatchY, caches)

          // update parameters
          if (optimizer == "gd") {
            parameters = updateParametersWithGd(parameters, grads, learningRate)
          } else if (optimizer == "momentum") {

            val pv = updateParametersWithMomentum(parameters,
                                                  grads,
                                                  v,
                                                  beta,
                                                  learningRate)
            parameters = pv._1
            v = pv._2
          } else if (optimizer == "adam") {
            val pvs = updateParametersWithAdam(parameters,
                                               grads,
                                               v,
                                               s,
                                               learningRate,
                                               beta1,
                                               beta2,
                                               epsilon)
            parameters = pvs._1
            v = pvs._2
            s = pvs._3
          }
      }

      if (printCost && i % 1000 == 0) {
        val (a3, caches) = forwardPropagation(x, parameters)
        cost = computeCost(a3, y)
        print(s"Cost after epoch $i: $cost ")
        val (accuracy, _) = OptUtils.predict(x, y, parameters)
        println(s"Accuracy: $accuracy")
      }
    }

    parameters

  }

}
