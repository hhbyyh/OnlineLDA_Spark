package com.github.yuhao.yang


import org.apache.spark.SparkContext

import breeze.linalg.{DenseVector => BDV, normalize, kron, sum, axpy => brzAxpy, DenseMatrix => BDM}
import breeze.numerics.{exp, abs, digamma}
import breeze.stats.distributions.Gamma


object OnlineLDA_Spark {

  /**
   * Online LDA can also be ran as batch LDA when submitting the corpus multiple times
   * Note that Batch LDA is just for illustration and the performance is not optimized. You may want to refactor the
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
    _K: Int,        // topic number
     D: Int,        // corpus size
    alpha: Double,  // Hyperparameter for prior on weight vectors theta
    eta: Double,    // Hyperparameter for prior on topics beta
    tau0: Double,   // downweights early iterations
    kappa: Double   // how quickly old infomation is forgotten
    ) extends Serializable {

  private var _updatect = 0
  private val _W = vocab.size

  // Initialize the variational distribution q(beta|lambda)
  var _lambda = getGammaMatrix(_K, _W)   // K * V
  private var _Elogbeta = dirichlet_expectation(_lambda)             // K * V
  private var _expElogbeta = exp(_Elogbeta)                               // K * V

  private def submit(sc: SparkContext, docPaths: Seq[String]): Unit = {
    // rhot will be between 0 and 1, and says how much to weight the information we got from this mini-batch.
    val rhot = math.pow(tau0 + _updatect, -kappa)

    val docsRdd = sc.parallelize(docPaths, 4)
    var stat = BDM.zeros[Double](_K, _W)
    stat = docsRdd.aggregate(stat)(seqOp, _ += _)

    stat = stat :* _expElogbeta
    _lambda = _lambda * (1 - rhot) + (stat * D.toDouble / docPaths.size.toDouble + eta) * rhot
    _Elogbeta = dirichlet_expectation(_lambda)
    _expElogbeta = exp(_Elogbeta)
    _updatect += 1

  }

  private def seqOp(other: BDM[Double], path: String): BDM[Double] = {
    val docs = scala.io.Source.fromFile(path).getLines()
    docs.foreach(doc => {
      val docVector = Docs2Vec.String2Vec(doc, vocab)
      val (ids, cts) = (docVector.indices.toList, docVector.values)

      var gammad = new Gamma(100, 1.0 / 100.0).samplesVector(_K).t   // 1 * K
      var Elogthetad = vector_dirichlet_expectation(gammad.t).t // 1 * K
      var expElogthetad = exp(Elogthetad.t).t // 1 * K
      val expElogbetad = _expElogbeta(::, ids).toDenseMatrix // K * ids

      var phinorm = expElogthetad * expElogbetad + 1e-100 // 1 * ids
      var meanchange = 1D
      val ctsVector = new BDV[Double](cts).t // 1 * ids

      while (meanchange > 1e-6) {
        val lastgamma = gammad
        //        1*K                  1 * ids               ids * k
        gammad = (expElogthetad :* ((ctsVector / phinorm) * (expElogbetad.t))) + alpha
        Elogthetad = vector_dirichlet_expectation(gammad.t).t
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


  def getGammaMatrix(row:Int, col:Int): BDM[Double] ={
      val gammaRandomGenerator = new Gamma(100, 1.0 / 100.0)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    (new BDM[Double](row, col, temp))
  }

  def dirichlet_expectation(alpha : BDM[Double]): BDM[Double] = {
    val rowSum =  sum(alpha(breeze.linalg.*, ::))
    val digAlpha = digamma(alpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }

  def vector_dirichlet_expectation(v : BDV[Double]): (BDV[Double]) ={
    digamma(v) - digamma(sum(v))
  }

}


