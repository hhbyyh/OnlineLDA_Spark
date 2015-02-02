/**
 * Created by yuhao on 1/28/15.
 */
package com.github.yuhao.yang
import org.apache.spark.mllib.linalg.{Vectors, SparseVector}
import org.apache.spark.{SparkContext}


object Docs2Vec extends Serializable{

  def defaultTokenizer = (doc: String) => doc.replaceAll("[^A-Za-z]+", " ").split("\\s+").filter(x => x.length > 2)

  def String2Vec(doc: String,
                 vocab: Map[String, Int],
                 tokenizer: (String) => Array[String] = defaultTokenizer): SparseVector = {

    require(vocab != null && vocab.size > 0, "empty vocabulary!")
    val tokens = tokenizer(doc)
    val groups = tokens.map(token => (vocab.getOrElse(token, -1), 1)).filter(_._1 > 0)
      .groupBy(_._1).mapValues(_.size.toDouble).toSeq
    Vectors.sparse(vocab.size, groups).asInstanceOf[SparseVector]
  }

  def extractVocab(sc:SparkContext,
                   paths: Seq[String],
                   stopWordsPath: String,
                   tokenizer: (String) => Array[String] = defaultTokenizer): Map[String, Int] = {

    val stopWords = (if (stopWordsPath != null) tokenizer(sc.textFile(stopWordsPath).collect().mkString(" "))
                    else Array[String]()).toSet

    val keys = paths.map(path => {
      val doc = sc.textFile(path).collect().mkString(" ")
      val tokens = doc.replaceAll("[^A-Za-z]+", " ").split("\\s+").filter(x => x.length > 2)
      tokens.distinct
    }).aggregate(Set[String]())((U, arr) => U ++ arr.toSet, _ ++ _)

    (keys -- stopWords).toArray.zipWithIndex.reverse.toMap
  }



}

