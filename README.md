# OnlineLDA_Spark
Online LDA based on Spark

The repository contains an implementation for online LDA from http://www.cs.princeton.edu/~mdhoffma/. 
Developed with Apache Spark v1.2.0. Share the code in case it may help.


main interfaces:

`OnlineLDA_Spark.runOnlineMode(sc: SparkContext, paths: Seq[String], vocab: Map[String, Int], K: Int, batchSize: Int)` and

`OnlineLDA_Spark runBatchMode(sc: SparkContext, paths: Seq[String], vocab: Map[String, Int], K: Int, iterations: Int)`

where paths are the files to be processed. For more details, refer to Driver.scala for examples.


Performance statistics, On a 4-node cluster, each with 16 cores and 30G memory. Without native BLAS installed. (for larger K, a native BLAS library will probably help)

1. data set from stackoverflow posts titles

    processed 8 millions short articles in 15 minutes (with K = 10 and vocab size about 110K).
    
2. full English wiki 
 
    processed 5876K documents (avg length ~1000 words/per doc, 30G in total) in 2 hours and 17 minutes. 
    word-topic sample:
        (album,score,music,song,band,seed,web,team,single,length)
        (align,center,style,left,small,text,width,background,class,color)
        (team,football,season,player,goals,players,goal,image,time,years)
        (year,news,publisher,years,time,books,book,state,government,work)
        (subdivision,population,area,image,map,display,code,footnotes,region,caption)
        (talk,span,color,font,page,style,discussion,deletion,small,made)
        (author,year,volume,issue,math,species,target,citations,system,publisher)
        (web,publisher,film,news,work,series,book,birth,author,language) ...

   
