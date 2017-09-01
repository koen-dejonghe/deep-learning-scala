package coursera

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import botkop.nn.coursera.MyNet
import botkop.nn.coursera.activations.{Identity, Relu, Sigmoid}
import numsca.Tensor
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import scala.io.Source

class MyNetSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  override def beforeAll(): Unit = {
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    numsca.rand.setSeed(213)
  }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def loadData(fname: String): (Tensor, Tensor) = {
    val (xDatas, yData) = Source
      .fromInputStream(gzis(fname))
      .getLines()
      .take(50)
      .foldLeft(Array.empty[Array[Double]], Array.empty[Double]) {
        case ((xs, ys), s) =>
          val tokens = s.split(",")
          val (y, x) =
            (tokens.head.toDouble, tokens.tail.map(_.toDouble / 255.0))
          (xs :+ x, ys :+ y)
      }

    val m = xDatas.length
    val n = xDatas.head.length

    val x = Tensor(xDatas.flatten).reshape(m, n).transpose
    val y = Tensor(yData).reshape(m, 1).transpose

    (x, y)
  }

  "MyNet" should "train mnist" in {

    val (x, y) = loadData("data/mnist_train.csv.gz")

    println(x.shape.toList)
    println(y.shape.toList)

    val layerDims = Array(x.shape(0), 100, 1)

    val activations =
      List.fill(layerDims.length - 1)(new Relu) :+ new Sigmoid

    val parameters = MyNet.lLayerModel(x,
                                       y,
                                       layerDims,
                                       numIterations = 10000,
                                       costFunction = MyNet.crossEntropyCost,
                                       activations = activations,
                                       learningRate = 1e-3,
                                       printCost = true)

  }

}
