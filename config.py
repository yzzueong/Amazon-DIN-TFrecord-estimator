import tensorflow.compat.v1 as tf

flags = tf.app.flags

#spark tfrecord path
#flags.DEFINE_string("data_gen_method", "spark", "data generate method")
#flags.DEFINE_string("train_record_dir", "xxx/train-tfrecord/", "train tfrecord dataset path")
#flags.DEFINE_string("test_record_dir", "xxx/test-tfrecord/", "test tfrecord dataset path")
#python tfrecord path
flags.DEFINE_string("data_gen_method", "python", "data generate method")
flags.DEFINE_string("train_record_dir", "xxx/train.tfrecords", "train tfrecord dataset path")
flags.DEFINE_string("test_record_dir", "xxx/test.tfrecords", "test tfrecord dataset path")
flags.DEFINE_string('model_dir', "./model_dir", "model dir to save checkpoint")
flags.DEFINE_integer('save_checkpoints_steps', 1000, "steps to save checkpoint")
flags.DEFINE_integer("batch_size",32, "batch size of training")
flags.DEFINE_boolean("run_on_cluster", False, "train_on_cluster")
flags.DEFINE_string("hidden_units", "80,40", "hidden units.")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_string("attention_hidden_units", "80,40", "hidden units.")
flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate")
flags.DEFINE_integer("train_steps", 3000000, "Number of (global) training steps to perform")
flags.DEFINE_integer("item_size", 63001, "Number of item size")
flags.DEFINE_integer("item_embedding_size", 128, "item's embedding size")
flags.DEFINE_integer("cate_size", 801, "Number of cate size")
flags.DEFINE_integer("cate_embedding_size", 128, "cate's embedding size")
flags.DEFINE_string("output_model", "./saved_model", "export model dir")


FLAGS = flags.FLAGS
