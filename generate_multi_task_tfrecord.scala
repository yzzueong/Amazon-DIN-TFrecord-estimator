import org.apache.spark.sql.{DataFrame, SaveMode}
import org.apache.spark.sql.functions.{col, collect_list, collect_set, lit, rank, udf}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.types._
import scala.util.Random

val revew_data = spark.read.option("header","true").option("sep","`").csv("./reviews.csv").select(col("reviewerID"), col("asin"), col("overall"), col("unixReviewTime").cast(IntegerType))//.orderBy(col("unixReviewTime"))

val meta_data = spark.read.option("header","true").option("sep","`").csv("./meta.csv")
//加索引
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
val need_idx = Array("asin", "categories")
val pipeline = new Pipeline().setStages(need_idx.map { colName =>
   new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(colName + "_idx")
}.toArray)
val indexed = pipeline.fit(meta_data).transform(meta_data)

val original_data = revew_data.join(indexed, Seq("asin"), "inner").select(col("reviewerID").alias("user_id"), col("asin_idx").cast(IntegerType).alias("item_id"), col("categories_idx").cast(IntegerType).alias("category_id"), col("overall").cast(IntegerType), col("unixReviewTime"))

val item_cate = indexed.select($"asin_idx".cast(IntegerType).alias("item_id"),col("categories_idx").cast(IntegerType).alias("pad_category_ids")).distinct()

val w = Window.partitionBy("user_id").orderBy("unixReviewTime")
val userbehaviorsequence_withneg = original_data
    .withColumn("hist", collect_list("item_id").over(w))
    .withColumn("hist_overall", collect_list("overall").over(w))
    .withColumn("hist_cate", collect_list("category_id").over(w))
    .groupBy("user_id").agg(
      max($"hist") as "hist",
      max($"hist_cate") as "hist_cate",
      max($"hist_overall") as "hist_overall"
    )

val seq_maxlen = 40

val all_data = userbehaviorsequence_withneg.rdd.flatMap(row => {
  val user_id = row.getAs[String]("user_id")
  val pos_list = row.getAs[Seq[Int]]("hist")
  val cate_list = row.getAs[Seq[Int]]("hist_cate")
  val overall_list = row.getAs[Seq[Int]]("hist_overall")
  val rnd = new Random()

  def gen_neg(hist: Seq[Int]) = {
      var neg_list = Seq[Int]()
      for(i <- hist){
        var randval = rnd.nextInt(63000)
        while (hist.contains(randval)){
            randval = rnd.nextInt(63000)
        }
        neg_list = neg_list:+randval
      }
    neg_list
    }
  val neg_list = gen_neg(pos_list)
  //user_id, item_list, cate_list, overall_list, item_id, label, flag
  //缺少cate_id, overall
  var all_data_temp = List[List[Any]]()

  for (i <- 1 until pos_list.length) {
    val hist_i = (List.fill(seq_maxlen)(0) ++ pos_list.slice(0, i).toSeq).takeRight(seq_maxlen)
    val hist_overall_i = (List.fill(seq_maxlen)(-1) ++ overall_list.slice(0, i).toSeq).takeRight(seq_maxlen)
    val hist_cate_i = (List.fill(seq_maxlen)(0) ++ cate_list.slice(0, i).toSeq).takeRight(seq_maxlen)
    // var hist_cate_i = gen_cate_list(hist_i)
    if (i == pos_list.length - 1) {
      all_data_temp = all_data_temp :+ List(user_id, hist_i, hist_cate_i, hist_overall_i, pos_list(i), 1, "test")
      all_data_temp = all_data_temp :+ List(user_id, hist_i, hist_cate_i, hist_overall_i, neg_list(i), 0, "test")
    } else {
      all_data_temp = all_data_temp :+ List(user_id, hist_i, hist_cate_i, hist_overall_i, pos_list(i), 1, "train")
      all_data_temp = all_data_temp :+ List(user_id, hist_i, hist_cate_i, hist_overall_i, neg_list(i), 0, "train")
    }
  }
  all_data_temp

}).map {case List(user_id:String, hist: Seq[Int], hist_cate: Seq[Int], hist_overall_i:Seq[Int], item_id:Int, label:Int, flag:String) => (user_id, hist, hist_cate, hist_overall_i, item_id, label, flag) }
  .toDF("user_id", "item_list", "cate_list", "overall_list", "item_id", "label", "flag")

val final_data = all_data.join(original_data.select($"item_id",$"category_id").distinct, Seq("item_id"), "left").join(original_data.select($"user_id",$"item_id",$"overall").groupBy("user_id","item_id").agg(max($"overall") as "overall"), Seq("user_id","item_id"), "left").na.fill(Map("overall" -> -1))


import org.apache.spark.sql.SaveMode
final_data.filter("flag='train'").withColumn("rand", rand()).orderBy(col("rand")).drop("flag", "rand").write.format("tfrecords").option("recordType", "Example").mode(SaveMode.Overwrite).save("./train-tfrecord")
final_data.filter("flag='test'").withColumn("rand", rand()).orderBy(col("rand")).drop("flag", "rand").write.format("tfrecords").option("recordType", "Example").mode(SaveMode.Overwrite).save("./test-tfrecord")