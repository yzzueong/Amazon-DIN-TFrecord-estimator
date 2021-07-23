import tensorflow.compat.v1 as tf
import os
import model
from feature import *
import config
import input_data
import json

FLAGS = config.FLAGS

tf.compat.v1.disable_eager_execution()

if FLAGS.run_on_cluster:
    cluster = json.loads(os.environ["TF_CONFIG"])
    task_index = int(os.environ["TF_INDEX"])
    task_type = os.environ["TF_ROLE"]

def main(unused_argv):
    feature_configs = FeatureConfig().create_features_columns()
    classifier = tf.estimator.Estimator(
        model_fn=model.build_base_model, #build_DIN_model
        config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=3),
        params = {
            "feature_configs": feature_configs,
            "hidden_units": FLAGS.hidden_units.split(","),
            "learning_rate": FLAGS.learning_rate,
            "attention_hidden_units": FLAGS.attention_hidden_units.split(','),
            "dropout_rate": FLAGS.dropout_rate
        }
    )
    
    def train_eval_model():
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_data.train_input_fn(FLAGS.train_record_dir, FLAGS.batch_size),
max_steps=FLAGS.train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_data.eval_input_fn(FLAGS.test_record_dir, FLAGS.batch_size),
                                          start_delay_secs=60,
                                          throttle_secs = 30,
                                          steps=200)
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    def export_model():
        feature_spec = feature_configs.feature_spec
        feature_map = {}
        for key, feature in feature_spec.items():
            #if key not in feature_configs.used_features:
            if key == "label":
                continue
            if isinstance(feature, tf.io.VarLenFeature):  # 可变长度
                feature_map[key] = tf.placeholder(dtype=feature.dtype, shape=[1], name=key)
            elif isinstance(feature, tf.io.FixedLenFeature):  # 固定长度
                feature_map[key] = tf.placeholder(dtype=feature.dtype, shape=[None, feature.shape[0]], name=key)
        serving_input_recevier_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        export_dir = classifier.export_saved_model(FLAGS.output_model, serving_input_recevier_fn)
 
    # 模型训练
    train_eval_model()
    
    # 导出模型，只在chief中导出一次即可
    if FLAGS.run_on_cluster: 
        if task_type == "chief":
            export_model()
    else:
        export_model()


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    #if FLAGS.run_on_cluster: parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)