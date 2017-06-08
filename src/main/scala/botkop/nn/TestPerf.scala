package botkop.nn

import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

object TestPerf extends App {

  (1 to 50000).foreach { i =>
    sigmoid(randn(764,30) dot randn(30, 10))
  }

}
