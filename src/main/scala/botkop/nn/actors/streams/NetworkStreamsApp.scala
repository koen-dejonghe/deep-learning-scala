package botkop.nn.actors.streams

import botkop.nn.Network.mnistData
import akka.stream._
import akka.stream.scaladsl._
import akka.{Done, NotUsed}
import akka.actor.ActorSystem
import akka.stream.stage.{GraphStage, GraphStageLogic, OutHandler}
import akka.util.ByteString
import akka.stream.stage._

import scala.concurrent._
import scala.concurrent.duration._
import org.nd4j.linalg.api.ndarray.{INDArray => Matrix}
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

case class Entry(x: Matrix, y: Matrix)

class Layer(weights: Matrix, biases: Matrix) {
  import Layer._

  def forward(x: Matrix): Matrix = sigmoid((weights dot x) + biases)

  def backward(activations: List[Matrix],
               layerIndex: Int,
               wb: Matrix): (Matrix, Matrix) = {
    val sp = derivative(activations(layerIndex))
    val delta = wb * sp
    (delta, delta dot activations(layerIndex - 1).transpose())
  }

}

object Layer {
  def apply(m: Int, n: Int): Layer =
    new Layer(weights = randn(n, m) / Math.sqrt(m), biases = randn(m, 1))

  def derivative(z: Matrix): Matrix = z * (-z + 1.0)
}

class NetworkStreamsApp extends App {

  implicit val system = ActorSystem("QuickStart")
  implicit val materializer = ActorMaterializer()

  val (trainingData, validationData, testData) = mnistData()

  val batch = trainingData.take(10).map { case (x, y) => Entry(x, y) }

  val source: Source[Entry, NotUsed] = Source(batch)

}

/*
  calculate activations of a layer
 */
class ForwardCalculator(m: Int, n: Int)
    extends GraphStage[FlowShape[Matrix, Matrix]] {

  val in: Inlet[Matrix] = Inlet[Matrix]("ForwardCalculator.in")
  val out: Outlet[Matrix] = Outlet[Matrix]("ForwardCalculator.out")

  override def shape: FlowShape[Matrix, Matrix] = FlowShape.of(in, out)

  override def createLogic(inheritedAttributes: Attributes): GraphStageLogic = new GraphStageLogic(shape) {

    val layer = Layer(m, n)

    setHandler(out, new OutHandler {
      override def onPull(): Unit = {
        pull(in)
      }
    })



  }

}
