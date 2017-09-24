package cs231n.assignment2

import numsca.Tensor

object Layers {

  /**
    *     Computes the forward pass for an affine (fully-connected) layer.
    *
    *     The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    *     examples, where each example x[i] has shape (d_1, ..., d_k). We will
    *     reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    *     then transform it to an output vector of dimension M.
    *
    *     Inputs:
    *     - x: An array containing input data, of shape (N, d_1, ..., d_k)
    *     - w: An array of weights, of shape (D, M)
    *     - b: An array of biases, of shape (M,)
    *
    *     Returns a tuple of:
    *     - out: output, of shape (N, M)
    *     - cache: (x, w, b)
    */
  def affineForward(x: Tensor,
                    w: Tensor,
                    b: Tensor): (Tensor, (Tensor, Tensor, Tensor)) = {
    val xs = x.reshape(x.shape(0), w.shape(0))
    val out = (xs dot w) + b
    val cache = (x, w, b)
    (out, cache)
  }

  /**
    *     Computes the backward pass for an affine layer.
    *
    *     Inputs:
    *     - dout: Upstream derivative, of shape (N, M)
    *     - cache: Tuple of:
    *     - x: Input data, of shape (N, d_1, ... d_k)
    *     - w: Weights, of shape (D, M)
    *
    *     Returns a tuple of:
    *     - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    *     - dw: Gradient with respect to w, of shape (D, M)
    *     - db: Gradient with respect to b, of shape (M,)
    */
  def affineBackward(
      dout: Tensor,
      cache: (Tensor, Tensor, Tensor)): (Tensor, Tensor, Tensor) = {

    val (x, w, _) = cache

    val mul = dout dot w.T
    val dx = mul.reshape(x.shape)

    val xs = x.reshape(x.shape(0), w.shape(0)).T
    val dw = xs dot dout

    val db = numsca.sum(dout, axis = 0)

    (dx, dw, db)
  }

  /**
    *     Computes the forward pass for a layer of rectified linear units (ReLUs).
    *
    *     Input:
    *     - x: Inputs, of any shape
    *
    *     Returns a tuple of:
    *     - out: Output, of the same shape as x
    *     - cache: x
    */
  def reluForward(x: Tensor): (Tensor, Tensor) = {
    val out = numsca.maximum(x, 0.0)
    val cache = x
    (out, cache)
  }

  /**
    *     Computes the backward pass for a layer of rectified linear units (ReLUs).
    *
    *     Input:
    *     - dout: Upstream derivatives, of any shape
    *     - cache: Input x, of same shape as dout
    *
    *     Returns:
    *     - dx: Gradient with respect to x
    */
  def reluBackward(dout: Tensor, cache: Tensor): Tensor = {
    val x = cache > 0.0
    dout * x
  }

  /**
    *     Computes the loss and gradient using for multiclass SVM classification.
    *
    *     Inputs:
    *     - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    *       class for the ith input.
    *     - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    *       0 <= y[i] < C
    *
    *     Returns a tuple of:
    *     - loss: Scalar giving the loss
    *     - dx: Gradient of the loss with respect to x
    */
  def svmLoss(x: Tensor, y: Tensor): (Double, Tensor) = {
    val n = x.shape(0)
    val correctClassScores = x(y)
    val margins = numsca.maximum(x - correctClassScores + 1.0, 0.0)
    margins.put(y, 0)
    val loss = numsca.sum(margins) / n

    val numPos = numsca.sum(margins > 0, axis = 1)
    val dx = numsca.zerosLike(x)
    dx.put(margins > 0, 1)
    dx.put(y, (ix, d) => d - numPos(ix.head))
    dx /= n

    (loss, dx)
  }

  /**
    *     Computes the loss and gradient for softmax classification.
    *
    *     Inputs:
    *     - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    *       class for the ith input.
    *     - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    *       0 <= y[i] < C
    *
    *     Returns a tuple of:
    *     - loss: Scalar giving the loss
    *     - dx: Gradient of the loss with respect to x
    */
  def softmaxLoss(x: Tensor, y: Tensor): (Double, Tensor) = {

    val shiftedLogits = x - numsca.max(x, axis = 1)
    val z = numsca.sum(numsca.exp(shiftedLogits), axis = 1)
    val logProbs = shiftedLogits - numsca.log(z)
    val probs = numsca.exp(logProbs)
    val n = x.shape(0)
    val loss = - numsca.sum(logProbs(y)) / n

    val dx = probs
    dx.put(y, _ - 1)
    dx /= n

    (loss, dx)
  }

}
