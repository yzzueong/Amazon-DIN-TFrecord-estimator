# coding:utf-8
import tensorflow.compat.v1 as tf
import config
import os
from feature import *

FLAGS = config.FLAGS

def parser_record(record, feature_configs=None):
    if feature_configs is None:
        feature_configs = FeatureConfig()
        feature_configs.create_features_columns()
        
    features = tf.io.parse_single_example(record, feature_configs.feature_spec)
    label = features["label"]
    return features, label

def train_input_fn(trainfilepath=None, batch_size=1):
    if trainfilepath is None:
        trainfilepath = FLAGS.train_record_dir
    if FLAGS.data_gen_method == "spark":
        filenames = [trainfilepath+name for name in os.listdir(trainfilepath) if name.startswith("part")]
        files = tf.data.Dataset.list_files(filenames)
    elif FLAGS.data_gen_method == "python":
        files = tf.data.Dataset.list_files([trainfilepath])
    else:
        raise "Unknown data_gen_method"
    # if FLAGS.run_on_cluster:
    #     files_all = []
    #     for f in filenames:
    #         files_all += tf.gfile.Glob(f)
    #     train_worker_num = len(FLAGS.worker_hosts.split(","))
    #     hash_id = FLAGS.task_index if FLAGS.job_name == "worker" else train_worker_num - 1
    #     files_shard = [files for i, files in enumerate(files_all) if i % train_worker_num == hash_id]
    #     dataset = tf.data.TFRecordDataset(files_shard)
    # else:
        #Local mode run below
    
    dataset = tf.data.TFRecordDataset(files)
    
    #dataset = dataset.shuffle(batch_size*10)
    dataset = dataset.map(parser_record, num_parallel_calls=8).repeat()
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def eval_input_fn(valfilepath=None, batch_size=1):
    if valfilepath is None:
        valfilepath = FLAGS.val_record_dir
    if FLAGS.data_gen_method == "spark":
        filenames = [valfilepath+name for name in os.listdir(valfilepath) if name.startswith("part")]
        files = tf.data.Dataset.list_files(filenames)
    elif FLAGS.data_gen_method == "python":
        files = tf.data.Dataset.list_files([valfilepath])
    else:
        raise "Unknown data_gen_method"
    #dataset = files.apply(tf.contrib.data.parallel_interleave(lambda filename: tf.data.TFRecordDataset(filename), buffer_output_elements=batch_size*20, cycle_length=10))
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parser_record, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    return dataset
