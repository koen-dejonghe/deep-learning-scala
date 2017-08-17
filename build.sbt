name := """deep-learning-scala"""

version := "1.0-SNAPSHOT"

lazy val root = project in file(".")

scalaVersion := "2.11.11"

classpathTypes += "maven-plugin"


libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.5.2"
libraryDependencies += "com.typesafe.akka" %% "akka-stream" % "2.5.2"



libraryDependencies += "org.nd4j" %% "nd4s" % "0.8.0"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0"
//libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0" % "0.8.0"
//libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.8.0"

libraryDependencies += "org.scalanlp" %% "breeze" % "0.13.2"
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.13.2"

// https://mvnrepository.com/artifact/org.scalatest/scalatest_2.11
libraryDependencies += "org.scalatest" % "scalatest_2.11" % "3.0.3" % "test"


//javaOptions in run += "-Dscala.concurrent.context.minThreads=8"
//javaOptions in run += "-Dscala.concurrent.context.maxThreads=16"
javaOptions in run += "-Xmx8g"
//
fork := true
