package coursera.deeplearning.course2.week2

import numsca._
import org.nd4j.linalg.factory.Nd4j

import scala.language.postfixOps
import scala.util.Random

object OptimizationModel {

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
  def updateParametersWithGd(parameters: Map[String, Tensor],
                             grads: Map[String, Tensor],
                             learningRate: Double): Map[String, Tensor] = {

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
                        seed: Int = 0) = {

    println(x.shape.toList)

    val m = x.shape(1)

    // val shuffledY = y(::, permutation).reshape(1, m)

    // val numCompleteMinibatches = math.floor(m / miniBatchSize)

  }

}
