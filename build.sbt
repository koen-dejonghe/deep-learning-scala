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

libraryDependencies += "org.scalanlp" %% "breeze" % "0.13.1"
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.13.1"

// fork := true
