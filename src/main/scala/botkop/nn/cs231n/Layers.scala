package botkop.nn.cs231n

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

object Layers {

  class LayerCache
  case class AffineCache(x: INDArray, w: INDArray, b: INDArray)
      extends LayerCache

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
  def affineForward(x: INDArray,
                    w: INDArray,
                    b: INDArray): (INDArray, AffineCache) = {
    val xs = x.reshape(x.shape()(0), w.shape()(0))
    val out = (xs mmul w) addRowVector b
    val cache = AffineCache(x, w, b)
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
  def affineBackward(dout: INDArray,
                     cache: AffineCache): (INDArray, INDArray, INDArray) = {
    import cache._

    val mul = dout mmul w.transpose()
    val dx = mul.reshape(x.shape(): _*)

    val xs = x.reshape(x.shape()(0), w.shape()(0)).transpose()
    val dw = xs mmul dout

    val db = Nd4j.sum(dout, 0)

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
  def reluForward(x: INDArray): (INDArray, INDArray) = {
    val out = Transforms.max(x, 0.0)
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
  def reluBackward(dout: INDArray, cache: INDArray): INDArray = {
    val x = cache.data().asDouble().map(d => if (d <= 0.0) 0.0 else 1.0)
    val dx = dout mul Nd4j.create(cache.shape, x)
    dx
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
  def svmLoss(x: INDArray, y: INDArray): (Double, INDArray) = {

    val n = x.shape()(0).toDouble
    val xData = x.data().asDouble()
    val yData = y.data().asInt()

    val xRows = xData.grouped(x.shape()(1))

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
          if (d > 0.0)
            1.0
          else
            0.0
        }
        dRow(correctIdx) -= np
        dRow.map(_ / n)
    }

    val dx = Nd4j.create(dxData).reshape(x.shape(): _*)
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
  def softmaxLoss(x: INDArray, y: INDArray): (Double, INDArray) = {

    val shiftedLogits = x subColumnVector Nd4j.max(x, 1)
    val z = Nd4j.sum(Transforms.exp(shiftedLogits), 1)
    val logProbs = shiftedLogits subColumnVector Transforms.log(z)
    val probs = Transforms.exp(logProbs)
    val n = x.shape()(0).toDouble
    val loss = logProbs
      .data()
      .asDouble()
      .grouped(x.shape()(1))
      .zip(y.data.asInt.iterator)
      .foldLeft(0.0) {
        case (acc, (row, index)) =>
          acc - row(index)
      } / n

    val dxData = probs.data.asDouble
      .grouped(x.shape()(1))
      .zip(y.data.asInt.iterator)
      .map {
        case (row, correctIndex) =>
          row(correctIndex) -= 1.0
          row.map(_ / n)
      }
      .toArray

    val dx = Nd4j.create(dxData).reshape(x.shape(): _*)
    (loss, dx)

  }
}
