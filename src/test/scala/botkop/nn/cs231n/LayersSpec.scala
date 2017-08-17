package botkop.nn.cs231n

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.scalatest.{FlatSpec, Matchers}
import collection.JavaConverters._


class LayersSpec extends FlatSpec with Matchers {

  "A Layer" should "correctly compute the affine forward pass" in {

    val numInputs = 2
    val inputShape = Seq(4, 5, 6)
    val outputDim = 3

    val inputSize = numInputs * inputShape.product
    val weightSize = outputDim * inputShape.product

    val x = Nd4j.linspace(-0.1, 0.5, inputSize).reshape(numInputs +: inputShape: _*)
    val w = Nd4j.linspace(-0.2, 0.3, weightSize).reshape(inputShape.product, outputDim)
    val b = Nd4j.linspace(-0.3, 0.1, outputDim)

    val out = Layers.affineForward(x, w, b)._1

    val correctData = Array(1.49834967, 1.70660132, 1.91485297, 3.25553199, 3.5141327, 3.77273342)
    val correctOut = Nd4j.create(correctData).reshape(numInputs, outputDim)

    val error = relError(out, correctOut)
    error should be < 1e-7

  }

  it should "correctly compute the affine backward pass" in {

    Nd4j.getRandom.setSeed(231)
    val x = Nd4j.randn(Array(10, 2, 3))
    val w = Nd4j.randn(6, 5)
    val b = Nd4j.randn(1, 5)
    val dout = Nd4j.randn(10, 5)

    val cache = Layers.affineForward(x, w, b)._2
    val derivatives = Layers.affineBackward(dout, cache)

    // println(derivatives)

    // test gradient on x
    val h = 1e-5

    val grad = Nd4j.zeros(x.shape(): _*)
    import org.nd4j.linalg.api.iter.NdIndexIterator
    val iter = new NdIndexIterator(x.shape(): _*)
    while(iter.hasNext) {
      val nextIndex: Array[Int] = iter.next
      val oldVal: Double = x.getDouble(nextIndex: _*)
      x.put(nextIndex, Nd4j.create(Array(oldVal + h)))
      val pos = Layers.affineForward(x, w, b)._1
      x.put(nextIndex, Nd4j.create(Array(oldVal - h)))
      val neg = Layers.affineForward(x, w, b)._1

      x.put(Nd4j.create(nextIndex), oldVal))

      grad.put(nextIndex, Nd4j.sum((pos sub neg) mul dout) div (2.0 * h))
    }

    val dxError = relError(grad, derivatives._1)
    println(dxError)
  }

  /**
    * returns relative error
    */
  def relError(x: INDArray, y: INDArray): Double =
    Nd4j.max(Transforms.abs(x sub y) div (Transforms.abs(x) add Transforms.abs(y))).data().asDouble()(0)


}
