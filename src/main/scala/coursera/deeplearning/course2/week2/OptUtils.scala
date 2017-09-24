package coursera.deeplearning.course2.week2

import numsca._

import scala.io.Source

object OptUtils {

  /**
  Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    */
  def sigmoid(x: Tensor): Tensor = 1 / (1 + exp(-x))

  /**
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    */
  def relu(x: Tensor): Tensor = maximum(x, 0)

  /**
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    b1 -- bias vector of shape (layer_dims[l], 1)
                    Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                    bl -- bias vector of shape (1, layer_dims[l])

    Tips:
    - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1].
    This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
    - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
    */
  def initializeParameters(layerDims: Array[Int]): Map[String, Tensor] =
    (1 until layerDims.length).foldLeft(Map.empty[String, Tensor]) {
      case (parameters, l) =>
        val w = randn(layerDims(l), layerDims(l - 1)) *
          math.sqrt(2.0 / layerDims(l - 1))
        val b = zeros(layerDims(l), 1)
        parameters ++ Seq(s"W$l" -> w, s"b$l" -> b)
    }

  /**
    Implement the cost function

    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3

    Returns:
    cost - value of the cost function
    */
  def computeCost(yHat: Tensor, y: Tensor): Double = {
    val m = y.shape(1)
    // val logprobs = multiply(-log(yHat), y) + multiply(-log(1 - yHat), 1 - y)
    // val cost = 1.0 / m * sum(logprobs)
    // cost.squeeze()

    val cost = (-y.dot(log(yHat).T) - (1 - y).dot(log(1 - yHat).T)) / m

    cost.squeeze()
  }

  /**
    Implements the forward propagation (and computes the loss) presented in Figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()

    Returns:
    loss -- the loss function (vanilla logistic loss)
    */
  def forwardPropagation(
      x: Tensor,
      parameters: Map[String, Tensor]): (Tensor, Map[String, Tensor]) = {

    // retrieve parameters
    val w1 = parameters("W1")
    val b1 = parameters("b1")
    val w2 = parameters("W2")
    val b2 = parameters("b2")
    val w3 = parameters("W3")
    val b3 = parameters("b3")

    // LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    val z1 = dot(w1, x) + b1
    val a1 = relu(z1)
    val z2 = dot(w2, a1) + b2
    val a2 = relu(z2)
    val z3 = dot(w3, a2) + b3
    val a3 = sigmoid(z3)

    val cache = Map(
      "z1" -> z1,
      "a1" -> a1,
      "W1" -> w1,
      "b1" -> b1,
      "z2" -> z2,
      "a2" -> a2,
      "W2" -> w2,
      "b2" -> b2,
      "z3" -> z3,
      "a3" -> a3,
      "W3" -> w3,
      "b3" -> b3
    )
    (a3, cache)
  }

  /**
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
    cache -- cache output from forward_propagation()

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables

    */
  def backwardPropagation(x: Tensor,
                          y: Tensor,
                          cache: Map[String, Tensor]): Map[String, Tensor] = {

    val m = x.shape(1)
    val dz3 = 1.0 / m * (cache("a3") - y)
    val dW3 = dot(dz3, cache("a2").T)
    val db3 = sum(dz3, axis = 1)

    val da2 = dot(cache("W3").T, dz3)
    val dz2 = multiply(da2, cache("a2") > 0)
    val dW2 = dot(dz2, cache("a1").T)
    val db2 = sum(dz2, axis = 1)

    val da1 = dot(cache("W2").T, dz2)
    val dz1 = multiply(da1, cache("a1") > 0)
    val dW1 = dot(dz1, x.T)
    val db1 = sum(dz1, axis = 1)

    val gradients = Map("dz3" -> dz3,
                        "dW3" -> dW3,
                        "db3" -> db3,
                        "da2" -> da2,
                        "dz2" -> dz2,
                        "dW2" -> dW2,
                        "db2" -> db2,
                        "da1" -> da1,
                        "dz1" -> dz1,
                        "dW1" -> dW1,
                        "db1" -> db1)

    gradients
  }

  /**
    This function is used to predict the results of a  n-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    */
  def predict(x: Tensor,
              y: Tensor,
              parameters: Map[String, Tensor]): (Double, Tensor) = {
    val m = x.shape(1)
    val n = parameters.size / 2

    val (probas, _) = forwardPropagation(x, parameters)

    // convert probas to 0/1 predictions
    val p = probas > 0.5

    val accuracy = sum(p == y) / m
    (accuracy, p)
  }

  def readData(fileName: String, shape: Array[Int]): Tensor = {
    val data = Source
      .fromFile(fileName)
      .getLines()
      .map(_.split(",").map(_.toDouble))
      .flatten
      .toArray
    Tensor(data).reshape(shape)
  }

  def loadData(): (Tensor, Tensor) = {
    val trainX = readData("data/coursera/moon/train_x.csv", Array(2, 300))
    val trainY = readData("data/coursera/moon/train_y.csv", Array(1, 300))
    (trainX, trainY)
  }

}
