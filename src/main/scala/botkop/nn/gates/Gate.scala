package botkop.nn.gates

import numsca.Tensor

class Gate {
}

case class AddGate(a: Tensor, b: Tensor) {
  val ldA = 1.0
  val ldB = 1.0

  def forward: Tensor = a + b
  def dA(dout: Tensor): Tensor = dout * ldA
  def dB(dout: Tensor): Tensor = dout * ldB
}

case class MulGate(a: Tensor, b: Tensor) {
  val ldA: Tensor = b
  val ldB: Tensor = a

  def forward: Tensor = a * b
  def dA(dout: Tensor): Tensor = dout * ldA
  def dB(dout: Tensor): Tensor = dout * ldB
}
