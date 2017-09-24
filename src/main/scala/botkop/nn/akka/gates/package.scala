package botkop.nn.akka

import numsca.Tensor

package object gates {

  case class Forward(x: Tensor, y: Tensor)
  case class Backward(dz: Tensor)
  case class Predict(x: Tensor)

  sealed trait Gate {
    def +(other: Gate): Network = Network(List(this, other))
    def *(i: Int): Network = Network(List.fill(i)(this))
  }

  case object Relu extends Gate {
    def name(layer: Int) = s"relu-gate-$layer"
  }

  case object Sigmoid extends Gate {
    def name(layer: Int) = s"sigmoid-gate-$layer"
  }

  case object Linear extends Gate {
    def name(layer: Int) = s"linear-gate-$layer"
  }

}
