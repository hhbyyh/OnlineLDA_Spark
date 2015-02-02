package com.github.yuhao.yang

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Gamma

object BLAS{
  def dirichlet_expectation(alpha : DenseMatrix[Double]): DenseMatrix[Double] = {
    val rowSum =  sum(alpha(*, ::))
    val digAlpha = digamma(alpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, *) - digRowSum
    result
  }

  def vector_dirichlet_expectation(v : DenseVector[Double]): (DenseVector[Double]) ={
    //psi(alpha) - psi(n.sum(alpha))
    digamma(v) - digamma(sum(v))
  }

  def getGammaMatrix(shape: Double, scale: Double, row:Int, col:Int): DenseMatrix[Double] ={
    val gammaRandomGenerator = new Gamma(shape, scale)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    (new DenseMatrix[Double](col, row, temp)).t
  }

  def getGammaVector(shape: Double, scale: Double, col:Int): Transpose[DenseVector[Double]] ={
    val gammaRandomGenerator = new Gamma(shape, scale)
    val temp = gammaRandomGenerator.sample(col).toArray
    (new DenseVector[Double](temp)).t
  }
}