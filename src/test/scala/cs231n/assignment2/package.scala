package cs231n

import numsca.Tensor

package object assignment2 {

  /**
    * Evaluate a numeric gradient for a function that accepts an array and returns an array.
    */
  def evalNumericalGradientArray(f: (Tensor) => Tensor,
                                 x: Tensor,
                                 df: Tensor,
                                 h: Double = 1e-5): Tensor = {
    val grad = numsca.zeros(x.shape)
    val it = numsca.nditer(x)
    while (it.hasNext) {
      val ix = it.next

      val oldVal = x(ix)

      x.put(ix, oldVal + h)
      val pos = f(x)

      x.put(ix, oldVal - h)
      val neg = f(x)

      x.put(ix, oldVal)
      val g = (numsca.sum((pos - neg) * df) / (2.0 * h)).squeeze()
      grad.put(ix, g)
    }
    grad
  }

  /**
   a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
   */
  def evalNumericalGradient(f: (Tensor) => Double,
                            x: Tensor,
                            h: Double = 0.00001): Tensor = {

    val grad = numsca.zeros(x.shape)
    val it = numsca.nditer(x)
    while (it.hasNext) {
      val ix = it.next

      val oldVal = x(ix)

      x.put(ix, oldVal + h)
      val pos = f(x)

      x.put(ix, oldVal - h)
      val neg = f(x)

      x.put(ix, oldVal)

      val g = (pos - neg) / (2.0 * h)
      grad.put(ix, g)
    }

    grad
  }

  /**
    * returns relative error
    */
  def relError(x: Tensor, y: Tensor): Double = {
    import numsca._
    val n = abs(x - y)
    val d = maximum(abs(x) + abs(y), 1e-8)
    max(n / d).squeeze()
  }

}
