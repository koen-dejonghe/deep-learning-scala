package botkop.nn.akka

import numsca.Tensor

package object gates {

  case class Forward(x: Tensor)
  case class Backward(dz: Tensor)
  case class Predict(x: Tensor)
}