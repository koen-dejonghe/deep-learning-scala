package numsca

import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class NumscaSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  "An array" should "slice correctly (1)" in {
    val v = array(Array(2, 4, 6), Array(3, 1))
    val i = array(Array(1), Array(1, 1))
    val d = v(i)
    d(0) should be (4.0)
  }

  it should "slice correctly (2)" in {
    val v = array(Array(2, 4, 6, 8), Array(2, 2))
    val i = array(Array(1), Array(1, 1))
    val d = v(i)
    println(d)
    val e = array(Array(6, 8), Array(1, 2))
    d.array should be (e.array)
  }

  it should "slice correctly (3)" in {
    val v = arange(9).reshape(3, 3)
    val i = array(Array(1), Array(1, 1))
    val d = v(i)(i)
    println(d)
    d(0) should be (4.0)
  }

  it should "slice correctly (4)" in {
    val v = arange(30).reshape(10, 3)
    // println(v)
    // println()
    val i1 = arange(v.shape(0))
    val i2 = array(Array(0, 1, 2, 0, 1, 2, 0, 1, 2, 0), Array(10, 1))
    // println(i)
    // println()
    val d = v(i1)(i2)
    println(d)
  }


}
