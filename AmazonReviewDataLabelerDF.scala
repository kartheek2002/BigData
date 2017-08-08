// Databricks notebook source
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions.{coalesce, col, monotonically_increasing_id}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector

import sys.process._

import scala.util.{Success, Try}
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

import java.util.Date
import java.text.SimpleDateFormat
import java.util.concurrent.TimeUnit
import java.util.Properties

import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._

// COMMAND ----------

// File name variables
var fileLink = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz"
var fileName = "reviews_Digital_Music_5.json.gz"
var tempFolder = "/tmp"
var fileStoreFolder = "/FileStore/reviews_Digital_Music_5"

// COMMAND ----------

// Download file and move it FileStore
dbutils.fs.mkdirs(fileStoreFolder) 

var result = dbutils.fs.ls(fileStoreFolder).toString

val src = ("file:" + tempFolder + "/" + fileName)
val dst = (fileStoreFolder + "/" + fileName)

if (!result.contains(fileName)) {
  result = (("ls " + tempFolder) !!).toString
  println(result)
  if (result.contains(fileName)) {
    dbutils.fs.cp(src.toString, dst.toString)
    println("Moved file to FileStore")
  }
  else {
    result = (("wget -q -P /tmp " + fileLink) !).toString
    
    if(result == "0") {
      dbutils.fs.cp(src.toString, dst.toString)
      println("Downloaded and moved file to FileStore")
    }
    else {
      println("Downloaded failed")
      System.exit(1)
    }
  }
}
else {
  println("File already in FileStore")
}

// COMMAND ----------

// Stanford CoreNLP method to remove stopwords and normalize (lemma)
val stopWords = Set("stopWord")

def plainTextToLemmas(text: String, stopWords: Set[String]): Seq[String] = {
  val props = new Properties()
  props.put("annotators", "tokenize, ssplit, pos, lemma")
  val pipeline = new StanfordCoreNLP(props)
  val doc = new Annotation(text.replaceAll("[,.!?:;]", ""))
  pipeline.annotate(doc)
  val lemmas = new ArrayBuffer[String]()
  val sentences = doc.get(classOf[SentencesAnnotation])
  for (sentence <- sentences; token <- sentence.get(classOf[TokensAnnotation])) {
    val lemma = token.get(classOf[LemmaAnnotation])
    if (lemma.length > 2 && !stopWords.contains(lemma)) {
      lemmas += lemma.toLowerCase
    }
  }
  lemmas
}

// COMMAND ----------

// read file to df amd add reviewID column with unique ID number for each row
var fullReviewsDF = spark.read.json(dst).withColumn("reviewID", monotonically_increasing_id())
var reviewsDF = fullReviewsDF.randomSplit(Array(0.25, 0.75))(0)
reviewsDF.printSchema()
//println(reviewsDF.count())
//reviewsDF.show()

// COMMAND ----------

val initLemma = reviewsDF.select("reviewID", "reviewText").as[(BigInt, String)].rdd
                    .map(r => (r._1, plainTextToLemmas(r._2.toString, stopWords)
                    .map(_.mkString("")).mkString(" ")))
                    .toDF("id", "reviewText")

val initToken = new Tokenizer().setInputCol("reviewText").setOutputCol("words")
val initReviewData = initToken.transform(initLemma)

val initHashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(5000)

val initFeaturizedData = initHashingTF.transform(initReviewData)
val initIDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val initIDFModel = initIDF.fit(initFeaturizedData)

val initData = initIDFModel.transform(initFeaturizedData).select("id", "features")
initData.printSchema()

// COMMAND ----------

val mpcModel = PipelineModel.load("/FileStore/deception_dataset/model/MultilayerPerceptronModel_withoutSymbolsStopWordsLemmaDF")

val newNames = Seq("id", "label")

val label_prediction = mpcModel.transform(initData).select("id", "prediction").toDF(newNames: _*)

label_prediction.printSchema()

//println(input.count())
//println(label_prediction.count())
//label_prediction.take(5).foreach(println)

// COMMAND ----------

// join labels to reviews
val reviewsLabeledDF = reviewsDF.join(label_prediction, col("reviewID") === col("id"))
                          .select($"reviewID",
                                  $"asin",
                                  $"helpful"(0) as "helpfulPos",
                                  $"helpful"(1) as "helpfulNeg",
                                  $"overall",
                                  $"reviewText",
                                  $"reviewTime",
                                  $"reviewerID",
                                  $"reviewerName",
                                  $"summary",
                                  $"unixReviewTime",
                                  $"label")

reviewsLabeledDF.printSchema()
//println(reviewsLabeledDF.filter("label == 0.0").count()/reviewsLabeledDF.count())
//reviewsLabeledDF.take(5).foreach(println)

// COMMAND ----------

reviewsLabeledDF.write.mode("overwrite").format("com.databricks.spark.csv").option("header", "true").save("/FileStore/LabeledData/" + fileStoreFolder.split("/")(2))

// COMMAND ----------


