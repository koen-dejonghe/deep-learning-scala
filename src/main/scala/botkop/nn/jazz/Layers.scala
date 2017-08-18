package botkop.nn.jazz

object Layers {

  def affineForward(x: Tensor, w: Tensor, b: Tensor) {
    val xs = x.reshape(x.shape(0), w.shape(0))

  }

}
