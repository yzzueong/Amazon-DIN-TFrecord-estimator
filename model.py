import feature as fe
from tensorflow.compat.v1 import feature_column as fc
import config
from tensorflow.python.estimator.canned import optimizers
import tensorflow.compat.v1 as tf


FLAGS = config.FLAGS

def attention(seq_emb, tid_emb, masks, params):
    max_seq_len = seq_emb.shape[1] # padded_dim
    embedding_size = FLAGS.item_embedding_size + FLAGS.cate_embedding_size
    u_emb = tf.reshape(seq_emb, shape=[-1, max_seq_len, embedding_size])
    #u_emb = seq_emb
    a_emb = tf.reshape(tf.tile(tid_emb, [1, max_seq_len]), shape=[-1, max_seq_len, embedding_size])
    net = tf.concat([u_emb, u_emb - a_emb, a_emb, u_emb * a_emb], axis=-1)
    for units in params['attention_hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    att_wgt = tf.layers.dense(net, units=1, activation=tf.nn.relu)
    att_wgt = tf.reshape(att_wgt, shape=[-1, 1, max_seq_len], name="weight")
    #wgt_emb = tf.multiply(seq_emb, att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
    #print(wgt_emb.shape)
    paddings = tf.ones_like(att_wgt) * (-2 ** 32 + 1)
    att_wgt = tf.where(masks, att_wgt, paddings)
    print(att_wgt.shape)
    att_wgt = att_wgt / (FLAGS.item_embedding_size)
    att_emb = tf.nn.softmax(att_wgt)
    att_emb = tf.squeeze(tf.matmul(att_emb, seq_emb), axis=1)
    #att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1, name="weighted_embedding")
    return att_emb

def build_DIN_model(features, labels, mode, params):
    feature_inputs = {}
    #DIN model features
    for key, value in params["feature_configs"].used_features.items():
        if key != 'label':
            feature_inputs[key] = fc.input_layer(features, value)
    
    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        item_embedding = tf.get_variable('item_embedding', [FLAGS.item_size, FLAGS.item_embedding_size])
        cate_embedding = tf.get_variable("cate_embedding", [FLAGS.cate_size, FLAGS.cate_embedding_size])
        inputs_seq_item_embedded = tf.nn.embedding_lookup(item_embedding, tf.cast(feature_inputs["item_list"], dtype=tf.int64))
        inputs_seq_cate_embedded = tf.nn.embedding_lookup(cate_embedding, tf.cast(feature_inputs["cate_list"], dtype=tf.int64))
        inputs_item_embedded = tf.nn.embedding_lookup(item_embedding, tf.cast(feature_inputs["item_id"], dtype=tf.int64))
        inputs_cate_embedded = tf.nn.embedding_lookup(cate_embedding, tf.cast(feature_inputs["cate_id"], dtype=tf.int64))
    seq_embedded = tf.concat([inputs_seq_item_embedded, inputs_seq_cate_embedded], axis=-1)
    target_embedded = tf.concat([inputs_item_embedded, inputs_cate_embedded], axis=-1)
    #target_embedded = tf.reshape(target_embedded, shape=(-1,target_embedded.shape[2]))
    #print("xxxxxxxx ",seq_embedded.shape, target_embedded.shape, inputs_item_embedded.shape, feature_inputs["cate_id"].shape)
    with tf.variable_scope("attention"):
        masks = tf.expand_dims(feature_inputs["item_list"] >= 0, axis=1)
        target_embedded = tf.reshape(target_embedded, shape=(-1,target_embedded.shape[2]))
        att_emb = attention(seq_embedded, target_embedded, masks, params)
    with tf.variable_scope("concat"):
        net = tf.layers.flatten(tf.concat([att_emb, target_embedded], axis=1))
    
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
            
    my_head = tf.estimator.BinaryClassHead()
    logits = tf.layers.dense(net, units=my_head.logits_dimension)
    
    optimizer = optimizers.get_optimizer_instance("Adam", params["learning_rate"])
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
      )

def build_base_model(features, labels, mode, params):
    feature_inputs = {}
    #DIN model features
    for key, value in params["feature_configs"].used_features.items():
        if key != 'label':
            feature_inputs[key] = fc.input_layer(features, value)
    
    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        item_embedding = tf.get_variable('item_embedding', [FLAGS.item_size, FLAGS.item_embedding_size])
        cate_embedding = tf.get_variable("cate_embedding", [FLAGS.cate_size, FLAGS.cate_embedding_size])
        inputs_seq_item_embedded = tf.nn.embedding_lookup(item_embedding, tf.cast(feature_inputs["item_list"], dtype=tf.int64))
        inputs_seq_cate_embedded = tf.nn.embedding_lookup(cate_embedding, tf.cast(feature_inputs["cate_list"], dtype=tf.int64))
        inputs_item_embedded = tf.nn.embedding_lookup(item_embedding, tf.cast(feature_inputs["item_id"], dtype=tf.int64))
        inputs_cate_embedded = tf.nn.embedding_lookup(cate_embedding, tf.cast(feature_inputs["cate_id"], dtype=tf.int64))

    seq_embedded = tf.layers.flatten(tf.concat([inputs_seq_item_embedded, inputs_seq_cate_embedded], axis=-1))
    target_embedded = tf.layers.flatten(tf.concat([inputs_item_embedded, inputs_cate_embedded], axis=-1))
    with tf.variable_scope("user_tower"):
        for units in params['hidden_units']:
            user_model = tf.layers.dense(seq_embedded, units=units, activation=tf.nn.relu)
            if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
                user_model = tf.layers.dropout(user_model, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
        user_model = tf.layers.dense(user_model, units=128, name="user_output_layer")

    with tf.variable_scope("item_tower"):
        for units in params['hidden_units']:
            item_model = tf.layers.dense(target_embedded, units=units, activation=tf.nn.relu)
            if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
                item_model = tf.layers.dropout(item_model, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
        item_model = tf.layers.dense(item_model, units=128, name="item_output_layer")

    dot = tf.multiply(user_model, item_model)
    my_head = tf.estimator.BinaryClassHead()
    logits = tf.layers.dense(dot, units=my_head.logits_dimension)
    
    optimizer = optimizers.get_optimizer_instance("Adam", params["learning_rate"])
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
      )
    