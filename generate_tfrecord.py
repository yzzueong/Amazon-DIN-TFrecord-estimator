import pandas as pd
import random
import numpy as np
import tensorflow.compat.v1 as tf

random.seed(1234)
#history click list max padding length
maxlen = 40

def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      i += 1
    df = pd.DataFrame.from_dict(df, orient='index')
    return df

reviews_df = to_df('../raw_data/reviews_Electronics_5.json')

meta_df = to_df('../raw_data/meta_Electronics.json')
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
meta_df = meta_df[['asin', 'categories']]
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key

asin_map, asin_key = build_map(meta_df, 'asin')
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')

user_count, item_count, cate_count, example_count =\
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
cate_list = np.array(cate_list, dtype=np.int32)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))]

  for i in range(1, len(pos_list)):
    hist = np.array(([0]*40 + pos_list[:i])[0-maxlen:])
    hist_cate = np.array([cate_list[idx] for idx in hist])
    
    if i != len(pos_list) - 1:
      train_set.append([ hist, hist_cate, pos_list[i], cate_list[pos_list[i]], 1])
      train_set.append([ hist, hist_cate, neg_list[i], cate_list[neg_list[i]], 0])
    else:
      #label = (pos_list[i], neg_list[i])
      #test_set.append((reviewerID, hist, label))
      test_set.append([ hist, hist_cate, pos_list[i], cate_list[pos_list[i]], 1])
      test_set.append([ hist, hist_cate, neg_list[i], cate_list[neg_list[i]], 0])

print("xxx", len(train_set))
print("xxxx", len(test_set))

random.shuffle(train_set)
random.shuffle(test_set)

train = pd.DataFrame(train_set, columns=['hist', 'hist_cate', 'item_id', 'pad_category_ids', 'label'])
test = pd.DataFrame(test_set, columns=['hist', 'hist_cate', 'item_id', 'pad_category_ids', 'label'])

writer=tf.python_io.TFRecordWriter(path="./train.tfrecords")
for idx in range(train.shape[0]):
    hist = [i for i in train["hist"][idx]]
    hist_cate = [i for i in train["hist_cate"][idx]]
    example=tf.train.Example(
        features=tf.train.Features(
            feature={
                "hist":tf.train.Feature(int64_list=tf.train.Int64List(value=hist)),
                "hist_cate":tf.train.Feature(int64_list=tf.train.Int64List(value=hist_cate)),
                "item_id":tf.train.Feature(int64_list=tf.train.Int64List(value=[train["item_id"][idx]])),
                "pad_category_ids":tf.train.Feature(int64_list=tf.train.Int64List(value=[train["pad_category_ids"][idx]])),
                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[train["label"][idx]]))
            }
        )
    )
    writer.write(record=example.SerializeToString())

writer.close()

writer=tf.python_io.TFRecordWriter(path="./test.tfrecords")
for idx in range(test.shape[0]):
    hist = [i for i in test["hist"][idx]]
    hist_cate = [i for i in test["hist_cate"][idx]]
    example=tf.train.Example(
        features=tf.train.Features(
            feature={
                "hist":tf.train.Feature(int64_list=tf.train.Int64List(value=hist)),
                "hist_cate":tf.train.Feature(int64_list=tf.train.Int64List(value=hist_cate)),
                "item_id":tf.train.Feature(int64_list=tf.train.Int64List(value=[test["item_id"][idx]])),
                "pad_category_ids":tf.train.Feature(int64_list=tf.train.Int64List(value=[test["pad_category_ids"][idx]])),
                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[test["label"][idx]]))
            }
        )
    )
    writer.write(record=example.SerializeToString())

writer.close()