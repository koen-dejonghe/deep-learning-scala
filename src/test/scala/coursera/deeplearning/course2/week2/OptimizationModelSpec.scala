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

  it should "update parameters with adam" in {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    /*
    {'W1': array([[ 1.62434536, -0.61175641, -0.52817175],
       [-1.07296862,  0.86540763, -2.3015387 ]]), 'b1': array([[ 1.74481176],
       [-0.7612069 ]]), 'W2': array([[ 0.3190391 , -0.24937038,  1.46210794],
       [-2.06014071, -0.3224172 , -0.38405435],
       [ 1.13376944, -1.09989127, -0.17242821]]), 'b2': array([[-0.87785842],
       [ 0.04221375],
       [ 0.58281521]])}
     */

    val w1 = Tensor(1.62434536, -0.61175641, -0.52817175, -1.07296862,
      0.86540763, -2.3015387).reshape(2, 3)
    val b1 = Tensor(1.74481176, -0.7612069).reshape(2, 1)
    val w2 = Tensor(0.3190391, -0.24937038, 1.46210794, -2.06014071, -0.3224172,
      -0.38405435, 1.13376944, -1.09989127, -0.17242821).reshape(3, 3)
    val b2 = Tensor(-0.87785842, 0.04221375, 0.58281521).reshape(3, 1)

    val parameters = Map("W1" -> w1, "b1" -> b1, "W2" -> w2, "b2" -> b2)

    /*
    {'dW1': array([[-1.10061918,  1.14472371,  0.90159072],
       [ 0.50249434,  0.90085595, -0.68372786]]), 'db1': array([[-0.12289023],
       [-0.93576943]]), 'dW2': array([[-0.26788808,  0.53035547, -0.69166075],
       [-0.39675353, -0.6871727 , -0.84520564],
       [-0.67124613, -0.0126646 , -1.11731035]]), 'db2': array([[ 0.2344157 ],
       [ 1.65980218],
       [ 0.74204416]])}
     */

    val dW1 = Tensor(-1.10061918, 1.14472371, 0.90159072, 0.50249434,
      0.90085595, -0.68372786).reshape(2, 3)
    val db1 = Tensor(-0.12289023, -0.93576943).reshape(2, 1)
    val dW2 =
      Tensor(-0.26788808, 0.53035547, -0.69166075, -0.39675353, -0.6871727,
        -0.84520564, -0.67124613, -0.0126646, -1.11731035).reshape(3, 3)
    val db2 = Tensor(0.2344157, 1.65980218, 0.74204416).reshape(3, 1)

    val grads = Map("dW1" -> dW1, "db1" -> db1, "dW2" -> dW2, "db2" -> db2)

    val (v, s) = OptimizationModel.initializeAdam(parameters)

    val (newParameters, newV, newS) =
      OptimizationModel.updateParametersWithAdam(parameters, grads, v, s)

    /*
    {'W1': array([[ 1.62750764, -0.61491869, -0.53133403],
       [-1.0761309 ,  0.86224535, -2.29837642]]), 'b1': array([[ 1.74797404],
       [-0.75804462]]), 'W2': array([[ 0.32220137, -0.25253265,  1.46527021],
       [-2.05697843, -0.31925493, -0.38089208],
       [ 1.13693172, -1.09672899, -0.16926593]]), 'b2': array([[-0.8810207 ],
       [ 0.03905147],
       [ 0.57965294]])}
     */
    val expectedW1 = Tensor(1.62750764, -0.61491869, -0.53133403, -1.0761309,
      0.86224535, -2.29837642).reshape(2, 3)
    assert(approxSameContents(newParameters("W1"), expectedW1))

    val expectedB1 = Tensor(1.74797404, -0.75804462).reshape(2, 1)
    assert(approxSameContents(newParameters("b1"), expectedB1))

    val expectedW2 =
      Tensor(0.32220137, -0.25253265, 1.46527021, -2.05697843, -0.31925493,
        -0.38089208, 1.13693172, -1.09672899, -0.16926593).reshape(3, 3)
    assert(approxSameContents(newParameters("W2"), expectedW2))

    val expectedB2 = Tensor(-0.8810207, 0.03905147, 0.57965294).reshape(3, 1)
    assert(approxSameContents(newParameters("b2"), expectedB2))

    /*
    {'dW1': array(-0.11006192,  0.11447237,  0.09015907,
       0.05024943,  0.09008559, -0.06837279),

     'dW2': array(-0.02678881,  0.05303555, -0.06916608,
      -0.03967535, -0.06871727, -0.08452056,
      -0.06712461, -0.00126646, -0.11173103),

     'db1': array(-0.01228902,
      -0.09357694),

     'db2': array( 0.02344157,
       0.16598022,
       0.07420442)
       }
     */

    val expectedVdW1 = Tensor(-0.11006192, 0.11447237, 0.09015907, 0.05024943,
      0.09008559, -0.06837279).reshape(2, 3)
    assert(approxSameContents(newV("dW1"), expectedVdW1))

    val expectedVdb1 = Tensor(-0.01228902, -0.09357694).reshape(2, 1)
    assert(approxSameContents(newV("db1"), expectedVdb1))

    val expectedVdW2 =
      Tensor(-0.02678881, 0.05303555, -0.06916608, -0.03967535, -0.06871727,
        -0.08452056, -0.06712461, -0.00126646, -0.11173103).reshape(3, 3)
    assert(approxSameContents(newV("dW2"), expectedVdW2))

    val expectedVdb2 = Tensor(0.02344157, 0.16598022, 0.07420442).reshape(3, 1)
    assert(approxSameContents(newV("db2"), expectedVdb2))

    /*
    {'dW1': array( 0.00121136,  0.00131039,  0.00081287,
        0.0002525 ,  0.00081154,  0.00046748),

        'dW2': array(  7.17640232e-05,   2.81276921e-04,   4.78394595e-04,
         1.57413361e-04,   4.72206320e-04,   7.14372576e-04,
         4.50571368e-04,   1.60392066e-07,   1.24838242e-03),

         'db1': array(  1.51020075e-05,
         8.75664434e-04),

         'db2': array(  5.49507194e-05,
         2.75494327e-03,
         5.50629536e-04)}
     */

    val expectedSdW1 = Tensor(0.00121136, 0.00131039, 0.00081287, 0.0002525,
      0.00081154, 0.00046748).reshape(2, 3)
    assert(approxSameContents(newS("dW1"), expectedSdW1))

    val expectedSdb1 = Tensor(1.51020075e-05, 8.75664434e-04).reshape(2, 1)
    assert(approxSameContents(newS("db1"), expectedSdb1))

    val expectedSdW2 = Tensor(7.17640232e-05, 2.81276921e-04, 4.78394595e-04,
      1.57413361e-04, 4.72206320e-04, 7.14372576e-04, 4.50571368e-04,
      1.60392066e-07, 1.24838242e-03).reshape(3, 3)
    assert(approxSameContents(newS("dW2"), expectedSdW2))

    val expectedSdb2 =
      Tensor(5.49507194e-05, 2.75494327e-03, 5.50629536e-04).reshape(3, 1)
    assert(approxSameContents(newS("db2"), expectedSdb2))

  }

  def approxSameContents(t1: Tensor,
                         t2: Tensor,
                         deviation: Double = 1e-8): Boolean =
    (t1 sameShape t2) && {
      val a = t1.array.dup.data.asDouble
      val b = t2.array.dup.data.asDouble
      //       println(a.toList)
      //       println(b.toList)
      !a.zip(b).exists {
        case (d1, d2) => math.abs(d2 - d1) > deviation
      }
    }

}
