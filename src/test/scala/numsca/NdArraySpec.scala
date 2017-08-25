package numsca

import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class NdArraySpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  "An NdArray" should "create correctly" in {
    assertThrows[Exception] {
      val shape = Array.empty[Int]
      val data = Array.empty[Double]
      new NdArray(shape, data)
    }

    assertThrows[Exception] {
      val shape = Array(1)
      val data = Array(1.0)
      new NdArray(shape, data)
    }

    assertThrows[Exception] {
      val shape = Array(2, 1)
      val data = Array(1.0)
      new NdArray(shape, data)
    }

    assertThrows[Exception] {
      val shape = Array(2, 1)
      val data = Array(1.0, 2.0, 3.0)
      new NdArray(shape, data)
    }

    val shape = Array(2, 1)
    val data = Array(1.0, 2.0)
    val a = new NdArray(shape, data)
    assert(shape sameElements a.shape)
    assert(data sameElements a.data)

  }

}
