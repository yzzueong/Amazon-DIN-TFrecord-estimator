#coding:utf-8
import tensorflow.compat.v1 as tf
from tensorflow import feature_column as fc
import config
 
FLAGS = config.FLAGS

class FeatureConfig(object):
    def __init__(self):
        self.feature_spec = dict()
        self.used_features = dict()
        
    def create_features_columns(self):
        #user behavior list
        item_list = fc.numeric_column(key="hist", shape=(40,), dtype=tf.int64)
        cate_list = fc.numeric_column(key="hist_cate", shape=(40,), dtype=tf.int64)
        
        #target item
        item_id = fc.numeric_column(key="item_id", dtype=tf.int64)
        cate_id = fc.numeric_column(key="pad_category_ids", dtype=tf.int64)
        
        #label
        label = fc.numeric_column(key="label",dtype=tf.int64)
        
        self.used_features = {
            "item_list": item_list,
            "cate_list": cate_list,
            "item_id": item_id,
            "cate_id": cate_id,
            "label": label
        }
        
        self.feature_spec = fc.make_parse_example_spec(self.used_features.values())
        return self