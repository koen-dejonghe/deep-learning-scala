package numsca

import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class TensorSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  "A tensor" should "self increment" in {
    val t = numsca.zeros(3, 3)
    t += 1

    for (i <- 0 until 3){
      for (j <- 0 until 3) {
        t(i, j) shouldBe 1.0
      }
    }
  }

  it should "evaluate gt" in {
    val t1 = numsca.zeros(2, 2)
    // val t2 = numsca.linspace(1.0, 4.0, Array(2, 2))
    val t2 = numsca.linspace(1.0, 4.0, 4).reshape(Array(2, 2))

    val c = t2 > 2
    println(c)

    val expected = Tensor(0, 0, 1, 1).reshape(2, 2)

    assert (
      c.array.dup().data().asDouble() sameElements expected.array.dup().data().asInt())
  }

  it should "broadcast addition" in {
    val t1 = numsca.linspace(1, 9, 9).reshape(Array(3,3))
    val t2 = numsca.ones(3, 1)

    val t3 = t1.array add t2.array.broadcast(t1.shape: _*)

    println(t3.data)
  }

  it should "slice" in {
    val a = arange(15).reshape(5, 3).array
    println(a)
    println()
    println(a.slice(2).slice(0))
    println()
    println(a.slices())

    val i = Tensor(0.0, 1.0, 0.0, 2.0, 1.0).reshape(1, 5)
    println(i)


  }



}
