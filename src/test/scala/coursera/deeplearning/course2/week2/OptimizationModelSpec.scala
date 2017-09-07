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
    val x = randn(12288, 148)
    val y = randn(1, 148) < 0.5
    (x, y, miniBatchSize)
  }

  "The model" should "create minibatches from a big batch" in {
    val (x, y, miniBatchSize) = randomMiniBatchesTestCase()

    val (xBatches, yBatches) =
      OptimizationModel.randomMiniBatches(x, y, miniBatchSize)

    xBatches.size shouldBe 3
    yBatches.size shouldBe 3

    xBatches.head.shape shouldBe Array(12288, 64)
    xBatches(1).shape shouldBe Array(12288, 64)
    xBatches(2).shape shouldBe Array(12288, 20)

    yBatches.head.shape shouldBe Array(1, 64)
    yBatches(1).shape shouldBe Array(1, 64)
    yBatches(2).shape shouldBe Array(1, 20)

    println(xBatches(1)(0 :> 1, 0 :> 1))
    println(yBatches(1)(0 :> 1, 0 :> 1))
  }

}
