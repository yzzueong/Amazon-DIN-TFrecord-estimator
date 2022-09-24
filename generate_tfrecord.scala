//amazon 数据集 scala spark代码, local, run at zeppelin
//save as tfrecord format

//need python data process before
// import pandas as pd
// def to_df(file_path):
//     """
//     转化为DataFrame结构
//     :param file_path: 文件路径
//     :return:
//     """
//     with open(file_path, 'r') as fin:
//         df = {}
//         i = 0
//         for line in fin:
//             df[i] = eval(line)
//             i += 1
//         df = pd.DataFrame.from_dict(df, orient='index')
//         return df
// reviews_df = to_df('xxx/reviews_Electronics_5.json')
// meta_df = to_df('xxx/meta_Electronics.json')[["asin","categories"]]
// meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
// meta_df = meta_df.reset_index(drop=True)
// meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
// reviews_df[["reviewerID","asin","unixReviewTime"]].to_csv("xxx/reviews.csv", sep="`", index=False)
// meta_df.to_csv("xxx/meta.csv", sep="`", index=False)

import org.apache.spark.sql.{DataFrame, SaveMode}
import org.apache.spark.sql.functions.{col, collect_list, collect_set, lit, rank, udf}
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.types._
import scala.util.Random
val revew_data = spark.read.option("header","true").option("sep","`").csv("xxx/reviews.csv").select(col("reviewerID"), col("asin"), col("unixReviewTime").cast(IntegerType))//.orderBy(col("unixReviewTime"))
val meta_data = spark.read.option("header","true").option("sep","`").csv("xxx/meta.csv")
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
val original_data = revew_data.join(indexed, Seq("asin"), "inner").select(col("reviewerID").alias("user_id"), col("asin_idx").cast(IntegerType).alias("item_id"), col("categories_idx").cast(IntegerType).alias("category_id"))

val item_cate = indexed.select($"asin_idx".cast(IntegerType).alias("item_id"),col("categories_idx").cast(IntegerType).alias("pad_category_ids")).orderBy("item_id").select("pad_category_ids").rdd.map(r => r.getAs[Int](0)).collect()

val userbehaviorsequence_withneg = original_data.groupBy("user_id").agg(
  collect_list("item_id") as "hist"
)

val seq_maxlen = 40
  
val all_data = userbehaviorsequence_withneg.rdd.flatMap(row => {
  val pos_list = row.getAs[Seq[Int]]("hist")
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
  
  def gen_cate_list(hist: Seq[Int]) = {
      var cate_list = Seq[Int]()
      for(i <- hist){
        cate_list = cate_list :+ item_cate(i)
      }
    cate_list
    }
  
  var all_data_temp = List[List[Any]]()

  for (i <- 1 until pos_list.length) {
    val hist_i = (List.fill(seq_maxlen)(item_cate(0)) ++ pos_list.slice(0, i).toSeq).takeRight(seq_maxlen)
    var hist_cate_i = gen_cate_list(hist_i)
    if (i == pos_list.length - 1) {
      all_data_temp = all_data_temp :+ List(hist_i, hist_cate_i, pos_list(i), item_cate( pos_list(i)), 1, "test")
      all_data_temp = all_data_temp :+ List(hist_i, hist_cate_i, neg_list(i), item_cate( neg_list(i)), 0, "test")
    } else {
      all_data_temp = all_data_temp :+ List(hist_i, hist_cate_i, pos_list(i), item_cate( pos_list(i)), 1, "train")
      all_data_temp = all_data_temp :+ List(hist_i, hist_cate_i, neg_list(i), item_cate( neg_list(i)), 0, "train")
    }
  }
  all_data_temp

}).map {case List(hist: Seq[Int], hist_cate: Seq[Int], item_id:Int, cate_id:Int, label:Int, flag:String) => (hist, hist_cate, item_id, cate_id, label, flag) }
  .toDF("hist", "hist_cate", "item_id", "pad_category_ids", "label", "flag")

import org.apache.spark.sql.SaveMode
all_data.filter("flag='train'").withColumn("rand", rand()).orderBy(col("rand")).drop("flag", "rand").write.format("tfrecords").option("recordType", "Example").mode(SaveMode.Overwrite).save("xxx/train-tfrecord")
all_data.filter("flag='test'").withColumn("rand", rand()).orderBy(col("rand")).drop("flag", "rand").write.format("tfrecords").option("recordType", "Example").mode(SaveMode.Overwrite).save("xxx/test-tfrecord")