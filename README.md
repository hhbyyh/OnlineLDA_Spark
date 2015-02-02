# OnlineLDA_Spark
Online LDA based on Spark

The repository contains an implementation for online LDA from http://www.cs.princeton.edu/~mdhoffma/. 
Developed with Apache Spark v1.2.0.

main interfaces:

OnlineLDA_Spark.runOnlineMode(sc: SparkContext, paths: Seq[String], vocab: Map[String, Int], K: Int, batchSize: Int) and
OnlineLDA_Spark runBatchMode(sc: SparkContext, paths: Seq[String], vocab: Map[String, Int], K: Int, iterations: Int)

where paths are the files to be processed. For more details, refer to Driver.scala for examples.

Performance statistics:

On a 4-node cluster, each with 16 cores and 30G memory.

1. data set from stackoverflow posts titles
    processed 8 millions short articles in 15 minutes (with K = 10 and vocab size about 110K).

