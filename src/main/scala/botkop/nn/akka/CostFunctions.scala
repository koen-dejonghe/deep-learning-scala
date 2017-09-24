package botkop.nn.akka

import numsca.Tensor


object CostFunctions {

  type CostFunction = (Tensor, Tensor) => (Double, Tensor)

  def crossEntropyCost(yHat: Tensor, y: Tensor): (Double, Tensor) = {
    val m = y.shape(1)
    val cost = (-y.dot(numsca.log(yHat).T) -
      (1 - y).dot(numsca.log(1 - yHat).T)) / m

    val dal = -(y / yHat - (1 - y) / (1 - yHat))
    // val dal = yHat - y
    (cost.squeeze(), dal)
  }

  def softmaxCost(xt: Tensor, yt: Tensor): (Double, Tensor) = {

    val x = xt.T
    val y = yt.T

    val shiftedLogits = x - numsca.max(x, axis = 1)
    val z = numsca.sum(numsca.exp(shiftedLogits), axis = 1)
    val logProbs = shiftedLogits - numsca.log(z)
    val probs = numsca.exp(logProbs)
    val n = x.shape(0)
    val loss = - numsca.sum(logProbs(y)) / n

    val dx = probs
    dx.put(y, _ - 1)
    dx /= n

    (loss, dx.T)
  }

}
