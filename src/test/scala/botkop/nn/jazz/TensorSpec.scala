package botkop.nn.jazz

import org.scalatest.{FlatSpec, Matchers}

class TensorSpec extends FlatSpec with Matchers {

  "A Tensor" should "get the correct element" in {
    val t = Tensor.zeros(2, 3)
    println(t(1, 2))
  }

  it should "correctly compute the dot product" in {
    val t1 = new Tensor(Array(1, 2, 3, 4, 5, 6), 2, 3)
    val t2 = new Tensor(Array(7, 8, 9), 3, 1)

    val d = t1 dot t2

    println(d.data.toList )
  }


}
