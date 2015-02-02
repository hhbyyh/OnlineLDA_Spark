package com.github.yuhao.yang

import java.util.Calendar
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import scala.collection.mutable.ArrayBuffer

object Driver extends Serializable{

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val inputDir = args(0)
    val filePaths = extractPaths(inputDir + "titles", true)
    val stopWordsPath = inputDir + "stop.txt"
    val vocabPath = inputDir + "wordsEn.txt"

    println("begin: " + Calendar.getInstance().getTime)
    println("path size: " + filePaths.size)
    assert(filePaths.size > 0)

    val conf = new SparkConf().setAppName("online LDA Spark")
    val sc = new SparkContext(conf)

    val vocab = Docs2Vec.extractVocab(sc, Seq(vocabPath), stopWordsPath)
    val vocabArray = vocab.map(_.swap)

    val K = args(1).toInt
//    val lda = OnlineLDA_Spark.runBatchMode(sc, filePaths, vocab, K, 50)
    val lda = OnlineLDA_Spark.runOnlineMode(sc, filePaths, vocab, K, args(2).toInt)

    println("_lambda:")
    for(row <- 0 until lda._lambda.rows){
      val v = lda._lambda(row, ::).t
      val topk = lda._lambda(row, ::).t.argtopk(10)
      val pairs = topk.map(k => (vocabArray(k), v(k)))
      val sorted = pairs.sortBy(_._2).reverse
       println(sorted.map(x => (x._1)).mkString(","), sorted.map(x => ("%2.2f".format(x._2))).mkString(","))
    }

    println("end: " + Calendar.getInstance().getTime())

  }

  def extractPaths(path: String, recursive: Boolean = true): Array[String] ={
    val docsets = ArrayBuffer[String]()
    val fileList = new java.io.File(path).listFiles()
    if(fileList == null) return docsets.toArray
    for(f <- fileList){
      if(f.isDirectory){
        if(recursive)
          docsets ++= extractPaths(f.getAbsolutePath, true)
      }
      else{
        docsets +=  f.getAbsolutePath
      }
    }
    docsets.toArray
  }

}
