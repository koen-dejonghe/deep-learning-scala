package botkop.nn.akka.cluster

import akka.actor.{ActorPath, ActorSystem}
import akka.cluster.client._
import akka.cluster.pubsub.{DistributedPubSub, DistributedPubSubMediator}
import botkop.nn.akka.CostFunctions._
import botkop.nn.akka.gates._
import botkop.nn.akka.optimizers._
import com.typesafe.config.ConfigFactory

object Main extends App {

  implicit val system: ActorSystem = ActorSystem("NeuralClusterSystem")

  val mediator = DistributedPubSub(system).mediator

  def optimizer() = Adam(learningRate = 0.0001)

  val (input, output) =
    ((Linear + Relu + Dropout) * 2 + Linear)
      .withDimensions(784, 100, 50, 10)
      .withOptimizer(optimizer)
      .withCostFunction(softmaxCost)
      .withRegularization(1e-5)
      .withMiniBatchSize(64)
      .init()

  println(output.path.toString)

  ClusterClientReceptionist.get(system).registerService(input)
  ClusterClientReceptionist.get(system).registerService(output)

  /*
  val receptionist = system.actorOf(
    ClusterReceptionist.props(mediator,
                              ClusterReceptionistSettings.create(system)))
                              */

}

object Client extends App {

  val config = ConfigFactory.parseString(
    """
     akka {
       actor {
         provider = "akka.remote.RemoteActorRefProvider"
       }

       remote {
         transport = "akka.remote.netty.NettyRemoteTransport"
         log-remote-lifecycle-events = off
         netty.tcp {
          hostname = "127.0.0.1"
          port = 5000
         }
       }
     }""")

  implicit val system: ActorSystem =
    ActorSystem("NeuralClientSystem", ConfigFactory.load(config))

  val initialContacts = Set(
    ActorPath.fromString(
      "akka.tcp://NeuralClusterSystem@127.0.0.1:2552/system/receptionist"))
  val settings = ClusterClientSettings(system)
    .withInitialContacts(initialContacts)

  val c = system.actorOf(ClusterClient.props(
                           ClusterClientSettings(system)
                           .withInitialContacts(initialContacts)
                         ),
                         "client")
  c ! ClusterClient.Send("/user/output", "hello!!!!!!!!!!!!!!!!!!!!!!!!!!!", localAffinity = true)
  c ! ClusterClient.SendToAll("/user/output", "hi!!!!!!!!!!!!!!!!!!!!!!!!!!!")

}
