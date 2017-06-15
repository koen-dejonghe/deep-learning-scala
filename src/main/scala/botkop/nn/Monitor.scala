package botkop.nn

import scala.collection.mutable.ListBuffer

case class Monitor(evaluationCost: ListBuffer[Double] = ListBuffer.empty,
                   evaluationAccuracy: ListBuffer[Double] = ListBuffer.empty,
                   trainingCost: ListBuffer[Double] = ListBuffer.empty,
                   trainingAccuracy: ListBuffer[Double] = ListBuffer.empty)

