package numsca

import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class TensorSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  "A tensor" should "self increment" in {
    val t = numsca.zeros(3, 3)
    t += 1

    for (i <- 0 until 3) {
      for (j <- 0 until 3) {
        t(i, j) shouldBe 1.0
      }
    }
  }

  it should "evaluate gt" in {
    val t1 = numsca.zeros(2, 2)
    val t2 = numsca.linspace(1.0, 4.0, 4).reshape(Array(2, 2))

    val c = t2 > 2

    val expected = Tensor(0, 0, 1, 1).reshape(2, 2)

    assert(c sameElements expected)
  }

  it should "broadcast addition" in {
    val t1 = numsca.linspace(1, 9, 9).reshape(Array(3, 3))
    val t2 = numsca.ones(3, 1)

    val t3 = t1.array add t2.array.broadcast(t1.shape: _*)

    println(t3.data)
  }

  it should "slice another tensor" in {
    val t1 = numsca.arange(15).reshape(5, 3)
    val t2 = Tensor(0, 1, 2, 0, 2).reshape(5, 1)
    val expected = Tensor(0.00, 4.00, 8.00, 9.00, 14.00).reshape(5, 1)
    val result = t1(t2)

    result.shape shouldBe expected.shape
    result.data shouldBe expected.data
  }

  it should "slice another tensor multi dim" in {
    val t1 = numsca.arange(24).reshape(4, 3, 2)
    val t2 = Tensor(1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1).reshape(4, 3, 1)

    val result = t1(t2)

    val expected =
      Tensor(1, 2, 5, 7, 8, 11, 13, 15, 16, 18, 20, 23).reshape(4, 3, 1)

    result.shape shouldBe expected.shape
    result.data shouldBe expected.data
  }

}
