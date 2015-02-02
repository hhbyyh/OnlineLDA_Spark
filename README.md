# OnlineLDA_Spark
Online LDA based on Spark

The repository contains an online version of LDA from http://www.cs.princeton.edu/~mdhoffma/ based on Spark.

main interfaces:

OnlineLDA_Spark.runOnlineMode(sc: SparkContext, paths: Seq[String], vocab: Map[String, Int], K: Int, batchSize: Int) and
 
OnlineLDA_Spark runBatchMode(sc: SparkContext, paths: Seq[String], vocab: Map[String, Int], K: Int, iterations: Int)

where paths are the files to be processed. For more details, refer to Driver.scala for examples.



