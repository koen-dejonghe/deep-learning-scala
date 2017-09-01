package botkop.nn.coursera.activations

import numsca.Tensor

trait Activation {
  type ForwardActivationFunction = Tensor => (Tensor, Tensor)
  type BackwardActivationFunction = (Tensor, Tensor) => Tensor
  def forward: ForwardActivationFunction
  def backward: BackwardActivationFunction
}

class Relu extends Activation {
  override def forward: ForwardActivationFunction = (z: Tensor) => {
    val a = numsca.maximum(0.0, z)
    (a, z)
  }

  override def backward: BackwardActivationFunction =
    (da: Tensor, cache: Tensor) => da * (cache > 0.0)
}

class Sigmoid extends Activation {
  override def forward: ForwardActivationFunction =
    (z: Tensor) => {
      val s = numsca.sigmoid(z)
      (s, s)
    }

  override def backward: BackwardActivationFunction =
    (da: Tensor, cache: Tensor) => da * cache * (-cache + 1)
}

class Identity extends Activation {
  override def forward: ForwardActivationFunction =
    (z: Tensor) => (z, z)
  override def backward: BackwardActivationFunction =
    (da: Tensor, _: Tensor) => da
}
