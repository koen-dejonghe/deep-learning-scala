name := """deep-learning-scala"""

version := "1.0-SNAPSHOT"

lazy val root = project in file(".")

scalaVersion := "2.11.11"

classpathTypes += "maven-plugin"


libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0"

libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.3" % "test"

libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.5.4"

// https://mvnrepository.com/artifact/com.typesafe.scala-logging/scala-logging_2.11
libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2"

// https://mvnrepository.com/artifact/ch.qos.logback/logback-classic
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3"




//javaOptions in run += "-Dscala.concurrent.context.minThreads=8"
//javaOptions in run += "-Dscala.concurrent.context.maxThreads=16"
javaOptions in run += "-Xmx8g"
fork := true
