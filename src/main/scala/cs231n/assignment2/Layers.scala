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

}
