package com.github.yuhao.yang

import org.apache.spark.SparkContext
import breeze.numerics._
import breeze.linalg._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib

object OnlineLDA_Spark {

  /**
   * Online LDA can also be ran as batch LDA when submitting the corpus multiple times
   * Note that Batch LDA is just for illustration and the performance is not optimaized. You may want to refactor the
   * submit method to improve performance.
   * @param paths one doc per line
   */
  def runBatchMode(sc: SparkContext, paths: Seq[String], vocab: Map[String, Int],
                   K: Int, iterations: Int): OnlineLDA_Spark = {
    val D = paths.size // one doc per path
    val olda = new OnlineLDA_Spark(vocab, K, D, 1.0 / K, 1.0 / K, 1024, 0)
    for (i <- 0 until iterations) {
      olda.submit(sc, paths)
      println(s"iteration $i of $iterations. Docs size: $D")
    }
    olda
  }


  /** Online LDA
    * Current implementation assumes each line of the file is a document.
    * @param paths one doc per line
    * @batchSize the size of the files per batch. Each line in a file is a document for LDA.
   */
  def runOnlineMode(sc: SparkContext, paths: Seq[String], vocab: Map[String, Int],
                    K: Int, batchSize: Int): OnlineLDA_Spark = {
    require(batchSize > 0 && K > 0 && paths.size > 0)
    val D = paths.size
    val lda = new OnlineLDA_Spark(vocab, K, D, 1.0 / K, 1.0 / K, 1024, 0.5)
    var batch = 0
    while (batch < paths.size) {
      val subPaths = paths.slice(batch, batch + batchSize)
      lda.submit(sc, subPaths)
      batch += batchSize
      println(s"batch $batch of " + paths.size)
    }
    lda
  }
}

class OnlineLDA_Spark(
    vocab: Map[String, Int], // vocabulary size
    _K: Int, // topic number
     D: Int, // corpus size
    alpha: Double, // Hyperparameter for prior on weight vectors theta
    eta: Double, // Hyperparameter for prior on topics beta
    tau0: Double, // downweights early iterations
    kappa: Double // how quickly old infomation is forgotten
    ) extends Serializable {

  private var _updatect = 0
  private val _W = vocab.size

  // Initialize the variational distribution q(beta|lambda)
  var _lambda = BLAS.getGammaMatrix(100.0, 1.0 / 100.0, _K, _W)   // K * V
  private var _Elogbeta = BLAS.dirichlet_expectation(_lambda)             // K * V
  private var _expElogbeta = exp(_Elogbeta)                               // K * V

  private def submit(sc: SparkContext, docPaths: Seq[String]): Unit = {
    // rhot will be between 0 and 1, and says how much to weight the information we got from this mini-batch.
    val rhot = pow(tau0 + _updatect, -kappa)

    val docsRdd = sc.parallelize(docPaths, 4)
    var stat = DenseMatrix.zeros[Double](_K, _W)
    stat = docsRdd.aggregate(stat)(seqOp, _ += _)

    stat = stat :* _expElogbeta
    _lambda = _lambda * (1 - rhot) + (stat * D.toDouble / docPaths.size.toDouble + eta) * rhot
    _Elogbeta = BLAS.dirichlet_expectation(_lambda)
    _expElogbeta = exp(_Elogbeta)
    _updatect += 1

  }

  private def seqOp(other: DenseMatrix[Double], path: String): DenseMatrix[Double] = {
    val docs = scala.io.Source.fromFile(path).getLines()
    docs.foreach(doc => {
      val docVector = Docs2Vec.String2Vec(doc, vocab)
      val (ids, cts) = (docVector.indices.toList, docVector.values.toList)

      var gammad = BLAS.getGammaVector(100, 1.0 / 100.0, _K) // 1 * K
      var Elogthetad = BLAS.vector_dirichlet_expectation(gammad.t).t // 1 * K
      var expElogthetad = exp(Elogthetad.t).t // 1 * K
      val expElogbetad = _expElogbeta(::, ids).toDenseMatrix // K * ids

      var phinorm = expElogthetad * expElogbetad + 1e-100 // 1 * ids
      var meanchange = 1D
      val ctsVector = new DenseVector[Double](cts.toArray).t // 1 * ids

      while (meanchange > 1e-6) {
        val lastgamma = gammad
        //        1*K                  1 * ids               ids * k
        gammad = (expElogthetad :* ((ctsVector / phinorm) * (expElogbetad.t))) + alpha
        Elogthetad = BLAS.vector_dirichlet_expectation(gammad.t).t
        expElogthetad = exp(Elogthetad.t).t
        phinorm = expElogthetad * expElogbetad + 1e-100
        meanchange = sum(abs((gammad - lastgamma).t)) / gammad.t.size.toDouble
      }

      val v1 = expElogthetad.t.toDenseMatrix.t
      val v2 = (ctsVector / phinorm).t.toDenseMatrix
      val outerResult = kron(v1, v2) // K * ids
      for (i <- 0 until ids.size) {
        other(::, ids(i)) := (other(::, ids(i)) + outerResult(::, i))
      }
    })
    other
  }
}


