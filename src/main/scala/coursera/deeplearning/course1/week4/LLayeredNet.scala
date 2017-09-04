package coursera.deeplearning.course1.week4

import numsca.Tensor

import scala.language.postfixOps

object LLayeredNet {

  class Cache
  class LinearCache(val a: Tensor, val w: Tensor, val b: Tensor) extends Cache
  class LinearActivationCache(val linearCache: LinearCache,
                              val activationCache: Tensor)
      extends Cache

  type ForwardActivationFunction = Tensor => (Tensor, Tensor)
  type BackwardActivationFunction = (Tensor, Tensor) => Tensor

  /**
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    */
  def initializeParameters(layerDims: Array[Int]): Map[String, Tensor] =
    (1 until layerDims.length).foldLeft(Map.empty[String, Tensor]) {
      case (parameters, l) =>
        val w = numsca.randn(layerDims(l), layerDims(l - 1)) / math.sqrt(layerDims(l-1))
        // val w = numsca.randn(layerDims(l), layerDims(l - 1)) * math.sqrt(2.0 / layerDims(l-1))
        val b = numsca.zeros(layerDims(l), 1)
        parameters ++ Seq(s"W$l" -> w, s"b$l" -> b)
    }

  /**
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    */
  def linearForward(a: Tensor, w: Tensor, b: Tensor): (Tensor, LinearCache) = {
    val z = w.dot(a) + b
    assert(z.shape sameElements Array(w.shape(0), a.shape(1)))
    val cache = new LinearCache(a, w, b)
    (z, cache)
  }

  /**
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    */
  def linearActivationForward(aPrev: Tensor,
                              w: Tensor,
                              b: Tensor,
                              activation: ForwardActivationFunction)
    : (Tensor, LinearActivationCache) = {
    val (z, linearCache) = linearForward(aPrev, w, b)
    val (a, activationCache) = activation(z)
    val cache = new LinearActivationCache(linearCache, activationCache)
    (a, cache)
  }

  /**
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently

    */
  def reluForward: ForwardActivationFunction = (z: Tensor) => {
    val a = numsca.maximum(z, 0.0)
    (a, z)
  }

  /**
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    */
  def reluBackward: BackwardActivationFunction =
    (da: Tensor, cache: Tensor) => {
      da * (cache > 0.0)
    }

  /**
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    */
  def sigmoidForward: ForwardActivationFunction =
    (z: Tensor) => {
      // small optimization compared to course: return the sigmoid as the cache,
      // since we can use it again in the back prop
      // this is the local derivative in Karpathy speak
      // (numsca.sigmoid(z), z)
      val s = numsca.sigmoid(z)
      (s, s)
    }

  /**
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z

    */
  def sigmoidBackward: BackwardActivationFunction =
    (da: Tensor, cache: Tensor) => {
      /*
      val z = cache
      val s = numsca.sigmoid(z)
      val dz = da * s * (-s + 1)
      dz
       */
      da * cache * (-cache + 1)
    }

  /**
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    */
  def lModelForward(x: Tensor, parameters: Map[String, Tensor])
    : (Tensor, List[LinearActivationCache]) = {
    val numLayers = parameters.size / 2

    (1 to numLayers).foldLeft(x, List.empty[LinearActivationCache]) {
      case ((aPrev, caches), l) =>
        val w = parameters(s"W$l")
        val b = parameters(s"b$l")
        val activation = if (l == numLayers) sigmoidForward else reluForward
        val (a, cache) = linearActivationForward(aPrev, w, b, activation)
        (a, caches :+ cache)
    }
  }

  /**
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    */
  def crossEntropyCost(yHat: Tensor, y: Tensor): Double = {
    val m = y.shape(1)

//    val logProbs = numsca.log(yHat) * y + (-y + 1) * numsca.log(-yHat + 1)
//    val cost = -numsca.sum(logProbs)(0, 0) / m
//    cost

    val cost = (-y.dot(numsca.log(yHat).transpose) - (-y + 1)
      .dot(numsca.log(-yHat + 1).transpose)) / m
    cost.squeeze()

  }

  /**
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    */
  def linearBackward(dz: Tensor,
                     cache: LinearCache): (Tensor, Tensor, Tensor) = {
    val aPrev = cache.a
    val w = cache.w
    val b = cache.b
    val m = aPrev.shape(1)

    val dw = dz.dot(aPrev.transpose) / m
    val db = numsca.sum(dz, axis = 1) / m
    val daPrev = w.transpose.dot(dz)

    assert(daPrev sameShape aPrev)
    assert(dw sameShape w)
    assert(db sameShape b)

    (daPrev, dw, db)
  }

  /**
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    */
  def linearActivationBackward(da: Tensor,
                               cache: LinearActivationCache,
                               backwardActivation: BackwardActivationFunction)
    : (Tensor, Tensor, Tensor) = {
    val dz = backwardActivation(da, cache.activationCache)
    val (daPrev, dw, db) = linearBackward(dz, cache.linearCache)

    (daPrev, dw, db)
  }

  /**
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...

    */
  def lModelBackward(
      al: Tensor,
      rawY: Tensor,
      caches: List[LinearActivationCache]): (Map[String, Tensor], Tensor) = {
    val numLayers = caches.size
    val y = rawY.reshape(al.shape)

    // derivative of cost with respect to AL
    val dal = -(y / al - (-y + 1) / (-al + 1))

    (1 to numLayers).reverse
      .foldLeft(Map.empty[String, Tensor], dal) {
        case ((grads, da), l) =>
          val currentCache = caches(l - 1)
          val activation =
            if (l == numLayers) sigmoidBackward else reluBackward
          val (daPrev, dw, db) =
            linearActivationBackward(da, currentCache, activation)
          val newGrads = grads ++ Seq(s"dA$l" -> daPrev,
                                      s"dW$l" -> dw,
                                      s"db$l" -> db)
          (newGrads, daPrev)
      }
  }

  /**
      Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    */
  def updateParameters(parameters: Map[String, Tensor],
                       grads: Map[String, Tensor],
                       learningRate: Double): Map[String, Tensor] =
    parameters.map {
      case (k, v) =>
        k -> { v - (grads(s"d$k") * learningRate) }
    }

  /**
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.

    */
  def lLayerModel(x: Tensor,
                  y: Tensor,
                  layerDims: Array[Int],
                  learningRate: Double = 0.0075,
                  numIterations: Int = 3000,
                  printCost: Boolean = false): Map[String, Tensor] = {

    val initialParameters = initializeParameters(layerDims)

    (1 to numIterations).foldLeft(initialParameters) {
      case (parameters, i) =>
        val (al, caches) = lModelForward(x, parameters)
        val cost = crossEntropyCost(al, y)
        if (printCost && i % 100 == 0) println(s"iteration $i: cost = $cost")
        val (grads, _) = lModelBackward(al, y, caches)
        updateParameters(parameters, grads, learningRate)
    }
  }

  /**
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X

    */
  def predict(x: Tensor, y: Tensor, parameters: Map[String, Tensor]): Double = {
    val m = x.shape(1)
    val n = parameters.size / 2

    val (probas, _) = lModelForward(x, parameters)

    // convert probas to 0/1 predictions
    val p = probas > 0.5

    val accuracy = numsca.sum(p == y) / m
    accuracy.squeeze()
  }

}
