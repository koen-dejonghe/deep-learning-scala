package botkop.nn.coursera

import botkop.nn.coursera.AndrewNet.{BackwardActivationFunction, ForwardActivationFunction}
import botkop.nn.coursera.activations.{Activation, Relu, Sigmoid}
import numsca.Tensor
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.language.postfixOps

object MyNet extends App {

  class Cache
  class LinearCache(val a: Tensor, val w: Tensor, val b: Tensor) extends Cache
  class LinearActivationCache(val linearCache: LinearCache,
                              val activationCache: Tensor)
      extends Cache

  type CostFunction = (Tensor, Tensor) => (Double, Tensor)

  def crossEntropyCost: CostFunction = (yHat: Tensor, y: Tensor) => {
    assert(
      yHat sameShape y,
      "y and yHat must have same shape (use one-hot encoding if necessary)")

    val m = y.shape(1)
    val logProbs = (y * numsca.log(yHat)) + ((-y + 1) * numsca.log(-yHat + 1))
    val cost = -numsca.sum(logProbs).squeeze() / m
    val dout = -(y / yHat - (-y + 1) / (-yHat + 1))

    (cost, dout)
  }

  def svmLoss: CostFunction = (x: Tensor, y: Tensor) => {

    val n = x.shape(0).toDouble
    val xData = x.array.dup.data.asDouble
    val yData = y.array.dup.data.asInt

    val xRows = xData.grouped(x.shape(1))

    val margins = xRows
      .zip(yData.iterator)
      .map {
        case (row, correctIndex) =>
          val correctScore = row(correctIndex)
          row.zipWithIndex.map {
            case (d, i) =>
              if (i == correctIndex)
                0.0
              else
                Math.max(0.0, d - correctScore + 1.0)
          }
      }
      .toArray

    val loss = margins.flatten.sum / n

    val numPos = margins.map { row =>
      row.count(_ > 0.0)
    }

    val dxData = margins.zipWithIndex.map {
      case (row, rowId) =>
        val correctIdx = yData(rowId)
        val np = numPos(rowId)
        val dRow: Array[Double] = row.map { d =>
          if (d > 0.0) 1.0 else 0.0
        }
        dRow(correctIdx) -= np
        dRow.map(_ / n)
    }

    val dx = Nd4j.create(dxData).reshape(x.shape: _*)
    (loss, new Tensor(dx))
  }

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
        // val w = numsca.randn(layerDims(l), layerDims(l - 1)) / math.sqrt(layerDims(l-1))
        val w = numsca.randn(layerDims(l), layerDims(l - 1)) * math.sqrt(
          2.0 / layerDims(l - 1))
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
  def lModelForward(
      x: Tensor,
      parameters: Map[String, Tensor],
      activations: List[Activation]): (Tensor, List[LinearActivationCache]) = {
    val numLayers = parameters.size / 2

    (1 to numLayers).foldLeft(x, List.empty[LinearActivationCache]) {
      case ((aPrev, caches), l) =>
        val w = parameters(s"W$l")
        val b = parameters(s"b$l")

        val activation = activations(l - 1).forward
        val (a, cache) = linearActivationForward(aPrev, w, b, activation)
        (a, caches :+ cache)
    }
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
      dout: Tensor,
      activations: List[Activation],
      caches: List[LinearActivationCache]): (Map[String, Tensor], Tensor) = {
    val numLayers = caches.size
    // val y = rawY.reshape(al.shape)

    (1 to numLayers).reverse
      .foldLeft(Map.empty[String, Tensor], dout) {
        case ((grads, da), l) =>
          val currentCache = caches(l - 1)
          val backward = activations(l - 1).backward
          val (daPrev, dw, db) =
            linearActivationBackward(da, currentCache, backward)
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
                  costFunction: CostFunction = crossEntropyCost,
                  activations: List[Activation] = List.empty,
                  printCost: Boolean = false): Map[String, Tensor] = {

    val initialParameters = initializeParameters(layerDims)

    val activationsToUse =
      if (activations.isEmpty)
        List.fill(layerDims.length - 1)(new Relu()) :+ new Sigmoid()
      else
        activations

    (1 to numIterations).foldLeft(initialParameters) {
      case (parameters, i) =>
        val (al, caches) = lModelForward(x, parameters, activationsToUse)
        val (cost, dout) = costFunction(al, y)
        if (printCost && i % 100 == 0) {
          println(s"iteration $i: cost = $cost")
          // println(numsca.sum(y == numsca.argmax(al)))
          println(numsca.sum(y == numsca.ceil(al)))
        }
        val (grads, _) = lModelBackward(al, y, dout, activationsToUse, caches)
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
  def predict(x: Tensor,
              y: Tensor,
              parameters: Map[String, Tensor],
              activations: List[Activation]): Double = {
    val m = x.shape(1)

    val (probas, _) = lModelForward(x, parameters, activations)

    // convert probas to 0/1 predictions
    val p = probas > 0.5

    val accuracy = numsca.sum(p == y) / m
    accuracy.squeeze()
  }

}
