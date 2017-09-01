package botkop.nn.dengue

import botkop.nn.coursera.AndrewNet._
import numsca.Tensor

import scala.io.Source

object Ingestor extends App {

  numsca.rand.setSeed(231)

  case class RawFeature(
      city: String,
      year: String,
      weekofyear: String,
      // week_start_date: String,
      // ndvi_ne: String,
      // ndvi_nw: String,
      // ndvi_se: String,
      // ndvi_sw: String,
      precipitation_amt_mm: String,
      reanalysis_air_temp_k: String,
      reanalysis_avg_temp_k: String,
      reanalysis_dew_point_temp_k: String,
      reanalysis_max_air_temp_k: String,
      reanalysis_min_air_temp_k: String,
      reanalysis_precip_amt_kg_per_m2: String,
      reanalysis_relative_humidity_percent: String,
      reanalysis_sat_precip_amt_mm: String,
      reanalysis_specific_humidity_g_per_kg: String,
      reanalysis_tdtr_k: String,
      station_avg_temp_c: String,
      station_diur_temp_rng_c: String,
      station_max_temp_c: String,
      station_min_temp_c: String,
      station_precip_mm: String
  ) {

    def toX: Option[Array[Double]] = {
      try {
        Some(
          Array(
            // ndvi_ne.toDouble,
            // ndvi_nw.toDouble,
            // ndvi_se.toDouble,
            // ndvi_sw.toDouble,
            precipitation_amt_mm.toDouble,
            reanalysis_air_temp_k.toDouble,
            reanalysis_avg_temp_k.toDouble,
            reanalysis_dew_point_temp_k.toDouble,
            reanalysis_max_air_temp_k.toDouble,
            reanalysis_min_air_temp_k.toDouble,
            reanalysis_precip_amt_kg_per_m2.toDouble,
            reanalysis_relative_humidity_percent.toDouble,
            reanalysis_sat_precip_amt_mm.toDouble,
            reanalysis_specific_humidity_g_per_kg.toDouble,
            reanalysis_tdtr_k.toDouble,
            station_avg_temp_c.toDouble,
            station_diur_temp_rng_c.toDouble,
            station_max_temp_c.toDouble,
            station_min_temp_c.toDouble,
            station_precip_mm.toDouble
          ))
      } catch {
        case _: Throwable => None
      }
    }

  }

  object RawFeature {
    def apply(s: String): RawFeature = {
      val a = s.replaceAll(",,", ", , ").replaceAll(",$", ", ").split(',')

      RawFeature(
        city = a(0),
        year = a(1),
        weekofyear = a(2),
        // week_start_date = a(3),
        // ndvi_ne = a(4),
        // ndvi_nw = a(5),
        // ndvi_se = a(6),
        // ndvi_sw = a(7),
        precipitation_amt_mm = a(8),
        reanalysis_air_temp_k = a(9),
        reanalysis_avg_temp_k = a(10),
        reanalysis_dew_point_temp_k = a(11),
        reanalysis_max_air_temp_k = a(12),
        reanalysis_min_air_temp_k = a(13),
        reanalysis_precip_amt_kg_per_m2 = a(14),
        reanalysis_relative_humidity_percent = a(15),
        reanalysis_sat_precip_amt_mm = a(16),
        reanalysis_specific_humidity_g_per_kg = a(17),
        reanalysis_tdtr_k = a(18),
        station_avg_temp_c = a(19),
        station_diur_temp_rng_c = a(20),
        station_max_temp_c = a(21),
        station_min_temp_c = a(22),
        station_precip_mm = a(23)
      )
    }
  }

  case class RawLabel(city: String,
                      year: String,
                      weekofyear: String,
                      total_cases: String) {
    def toY: Double = total_cases.toDouble
  }

  object RawLabel {
    def apply(s: String): RawLabel = {
      val a = s.split(",")
      RawLabel(a(0), a(1), a(2), a(3))
    }

  }

  def readTrainingFeatures(): List[RawFeature] =
    readData("data/dengue_features_train.csv", RawFeature.apply)

  def readTrainingLabels(): List[RawLabel] =
    readData("data/dengue_labels_train.csv", RawLabel.apply)

  def readData[T](filename: String, f: (String) => T): List[T] = {
    Source
      .fromFile(filename)
      .getLines()
      .zipWithIndex
      .flatMap {
        case (s, i) =>
          if (i == 0) None else Some(f(s))
      }
      .toList
  }

  val trainingFeatures = readTrainingFeatures()

  val groupedFeatures = trainingFeatures.groupBy { f =>
    (f.city, f.year, f.weekofyear)
  }
  // check that there's only 1 case per key
  assert(!groupedFeatures.values.exists(_.size != 1))

  val trainingLabels = readTrainingLabels()

  val groupedLabels = trainingLabels.groupBy { l =>
    (l.city, l.year, l.weekofyear)
  }
  // check that there's only 1 case per key
  assert(!groupedLabels.values.exists(_.size != 1))

  // check that sequence of keys for x and y are identical
  assert(groupedFeatures.keys.toList.sorted == groupedLabels.keys.toList.sorted)

  val allX = trainingFeatures.map(_.toX)
  val allY = trainingLabels.map(_.toY)

  val usableXY = allX.zip(allY).flatMap {
    case (maybeX, y) =>
      maybeX match {
        case None => None
        case Some(x) =>
          Some(x, y)
      }
  }

  println(s"usable entries: ${usableXY.size} out of: ${allX.size}")

  /*
  usableXY.map(_._1).foldLeft(0.0) { case (sums, a) =>

      ???
  }
   */

  val m = usableXY.size
  val n = usableXY.head._1.length

  val xData = usableXY.flatMap(_._1).toArray

  val x = numsca.create(xData, m, n).transpose
  println(x.shape.toList)

  // val normalized = scale(zeroCenter(x, 1), 1, -1.0, 1.0)
  val normalized = scale(x, 1)
  println(normalized)

  val yData: Array[Double] = usableXY.map(_._2).toArray
  // val y = normalize(numsca.create(yData, 1, m), 1)
  // val y = scale(numsca.create(yData, 1, m), 1)
  val y = numsca.create(yData, 1, m)

  val layerDims = Array(n, 20, 1)
  val pars = model(normalized,
                   y,
                   layerDims,
                   learningRate = 1e-8,
                   numIterations = 10000,
                   printCost = true)

  val (yHat, _) = modelForward(normalized, pars)
  println(yHat)
  println(y)


  def model(x: Tensor,
            y: Tensor,
            layerDims: Array[Int],
            learningRate: Double = 0.0075,
            numIterations: Int = 3000,
            printCost: Boolean = false): Map[String, Tensor] = {

    val initialParameters = initializeParameters(layerDims)

    (1 to numIterations).foldLeft(initialParameters) {
      case (parameters, i) =>
        val (al, caches) = modelForward(x, parameters)
        val cost = crossEntropyCost(al, y)
        // val cost = rmse(al, y)
        if (printCost && i % 100 == 0) {
          println(s"iteration $i: cost = $cost")
          println(al)
          println(y)
        }
        val (grads, _) = modelBackward(al, y, caches)
        updateParameters(parameters, grads, learningRate)
    }
  }

  def rmse(yHat: Tensor, y: Tensor): Double = {
    val m = y.shape(1)
    val sum = numsca.sum(numsca.power(yHat - y, 2))(0, 0)
    math.sqrt(sum / m)
  }

  def identityForward: ForwardActivationFunction = (z: Tensor) => (z, z)

  def identityBackward: BackwardActivationFunction =
    (da: Tensor, cache: Tensor) => da

  def modelForward(x: Tensor, parameters: Map[String, Tensor])
    : (Tensor, List[LinearActivationCache]) = {
    val numLayers = parameters.size / 2

    (1 to numLayers).foldLeft(x, List.empty[LinearActivationCache]) {
      case ((aPrev, caches), l) =>
        val w = parameters(s"W$l")
        val b = parameters(s"b$l")
        val activation = if (l == numLayers) sigmoidForward else reluForward
        // val activation = reluForward
        val (a, cache) = linearActivationForward(aPrev, w, b, activation)
        (a, caches :+ cache)
    }
  }

  def modelBackward(
      al: Tensor,
      rawY: Tensor,
      caches: List[LinearActivationCache]): (Map[String, Tensor], Tensor) = {
    val numLayers = caches.size
    val y = rawY.reshape(al.shape)

    // derivative of cost with respect to AL
    val dal = -(y / al - (-y + 1) / (-al + 1))
    // note: this is dout, the derivative of the loss function with respect to al

    // println("wwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
    // println(y.shape.toList)
    // println(dal.shape.toList)
    assert(dal sameShape y)

    (1 to numLayers).reverse
      .foldLeft(Map.empty[String, Tensor], dal) {
        case ((grads, da), l) =>
          val currentCache = caches(l - 1)
          val activation =
            // reluBackward
            if (l == numLayers) sigmoidBackward else reluBackward
          val (daPrev, dw, db) =
            linearActivationBackward(da, currentCache, activation)
          val newGrads = grads + (s"dA$l" -> daPrev) + (s"dW$l" -> dw) + (s"db$l" -> db)
          (newGrads, daPrev)
      }
  }

  def normalize(x: Tensor, axis: Int): Tensor = {
    val mean = new Tensor(x.array.mean(axis))
    val std = new Tensor(x.array.std(axis))
    val zeroCentered = x - mean
    val norm = zeroCentered / std
    norm
  }

  def zeroCenter(x: Tensor, axis: Int): Tensor = {
    val mean = new Tensor(x.array.mean(axis))
    val zeroCentered = x - mean
    zeroCentered
  }

  def scale(x: Tensor, axis: Int, from: Double = 0.0, until: Double = 1.0): Tensor = {

    val max = new Tensor(x.array.max(axis))
    val min = new Tensor(x.array.min(axis))

    val num = (x - min) * (until - from)
    val den = max - min
    val scaled = (num / den) + from

    scaled
  }

  def initializeParameters(layerDims: Array[Int]): Map[String, Tensor] =
    (1 until layerDims.length).foldLeft(Map.empty[String, Tensor]) {
      case (parameters, l) =>
        // val w = numsca.randn(layerDims(l), layerDims(l - 1)) / math.sqrt(layerDims(l-1))
        // val w = numsca.randn(layerDims(l), layerDims(l - 1)) * math.sqrt(2.0 / layerDims(l-1))
        val w = numsca.randn(layerDims(l), layerDims(l - 1)) * 0.01
        val b = numsca.zeros(layerDims(l), 1)
        parameters ++ Seq(s"W$l" -> w, s"b$l" -> b)
    }


}
