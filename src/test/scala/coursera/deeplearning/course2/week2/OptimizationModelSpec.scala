package coursera.deeplearning.course2.week2

import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.scalatest.{BeforeAndAfterEach, FlatSpec, Matchers}

import numsca._

class OptimizationModelSpec
    extends FlatSpec
    with Matchers
    with BeforeAndAfterEach {

  override def beforeEach(): Unit = {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    numsca.rand.setSeed(231)
  }

  def randomMiniBatchesTestCase(): (Tensor, Tensor, Int) = {
    val miniBatchSize = 64
    val x = randn(10, 10)
    val y = randn(1, 148) < 0.5
    (x, y, miniBatchSize)
  }

  "The model" should "create minibatches from a big batch" in {
    val (x, y, miniBatchSize) = randomMiniBatchesTestCase()

    val z = x(:>, 0 :> 10)

  }

}
