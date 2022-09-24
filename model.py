import feature as fe
from tensorflow.compat.v1 import feature_column as fc
import config
from tensorflow.python.estimator.canned import optimizers
import tensorflow.compat.v1 as tf
from tensorflow.keras.backend import repeat_elements, sum


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
    # print(att_wgt.shape)
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
        labels=labels["label"] if labels else None,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
      )

def build_base_model(features, labels, mode, params):
    feature_inputs = {}
    #DIN model features
    for key, value in params["feature_configs"].used_features.items():
        if key not in ["label", "overall"]:
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

    # print("xxxxxx", seq_embedded.shape, target_embedded.shape)
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
        labels=labels["label"] if labels else None,
        logits=logits,
        train_op_fn=lambda loss: optimizer.minimize(loss, global_step=tf.train.get_global_step())
      )

def MMoE(features, labels, mode, params):
    feature_inputs = {}
    # DIN model features
    for key, value in params["feature_configs"].used_features.items():
        if key not in ["label", "overall"]:
            feature_inputs[key] = fc.input_layer(features, value)

    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        item_embedding = tf.get_variable('item_embedding', [FLAGS.item_size, FLAGS.item_embedding_size])
        cate_embedding = tf.get_variable("cate_embedding", [FLAGS.cate_size, FLAGS.cate_embedding_size])
        inputs_seq_item_embedded = tf.nn.embedding_lookup(item_embedding,
                                                          tf.cast(feature_inputs["item_list"], dtype=tf.int64))
        inputs_seq_cate_embedded = tf.nn.embedding_lookup(cate_embedding,
                                                          tf.cast(feature_inputs["cate_list"], dtype=tf.int64))
        inputs_item_embedded = tf.nn.embedding_lookup(item_embedding,
                                                      tf.cast(feature_inputs["item_id"], dtype=tf.int64))
        inputs_cate_embedded = tf.nn.embedding_lookup(cate_embedding,
                                                      tf.cast(feature_inputs["cate_id"], dtype=tf.int64))


    # 加入用户历史评价作为权重
    # print("----------", tf.expand_dims(feature_inputs["overall_list"], axis=2).shape)
    inputs_seq_item_embedded_with_weight = inputs_seq_item_embedded * tf.expand_dims(feature_inputs["overall_list"], axis=2)
    inputs_seq_cate_embedded_with_weight = inputs_seq_cate_embedded * tf.expand_dims(feature_inputs["overall_list"], axis=2)


    seq_embedded = tf.concat([inputs_seq_item_embedded_with_weight, inputs_seq_cate_embedded_with_weight], axis=-1)
    target_embedded = tf.concat([inputs_item_embedded, inputs_cate_embedded], axis=-1)
    # target_embedded = tf.reshape(target_embedded, shape=(-1,target_embedded.shape[2]))
    # print("xxxxxxxx ",seq_embedded.shape, target_embedded.shape, inputs_item_embedded.shape, feature_inputs["cate_id"].shape)
    with tf.variable_scope("attention"):
        masks = tf.expand_dims(feature_inputs["item_list"] >= 0, axis=1)
        target_embedded = tf.reshape(target_embedded, shape=(-1, target_embedded.shape[2]))
        att_emb = attention(seq_embedded, target_embedded, masks, params)
    with tf.variable_scope("concat"):
        net = tf.layers.flatten(tf.concat([att_emb, target_embedded], axis=1))

    expert_outputs = []
    gate_outputs = []
    final_outputs = []
    logits = []
    with tf.variable_scope("MMoE"):
        with tf.variable_scope("expert_net"):
            for i in range(params["expert_net_num"]):
                expert_outputs.append(tf.expand_dims(
                    tf.layers.dense(net, params["units"], activation="relu", use_bias=True), axis=2
                ))
        with tf.variable_scope("gate_net"):
            for i in range(2):
                gate_outputs.append(tf.layers.dense(
                    net, params["expert_net_num"], activation="softmax", use_bias=True
                ))
        expert_output = tf.concat(expert_outputs, axis=2)

        for gate_output in gate_outputs:
            expand_gate_out = tf.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_output * repeat_elements(expand_gate_out, params["units"], axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))

        for index, task_layer in enumerate(final_outputs):
            tower_layer = tf.layers.dense(task_layer, units=8, activation="relu")
            output_layer = tf.layers.dense(tower_layer, units=1, activation=None)
            # logits.append(tf.squeeze(output_layer, axis=1))
            logits.append(output_layer)

    preds = [tf.sigmoid(logits[0]), logits[1]]

    predictions = {
        "label": preds[0],
        "overall": preds[1]
    }

    labels = [labels["label"], labels["overall"]] if labels else None

    export_output = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
    }

    # print("YYYYYYYYYYYYYYYYYYYYYYY",logits[0].shape, labels[0].shape)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_output)
    else:
        ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[0], labels=tf.cast(labels[0], tf.float32)))
        overall_loss = tf.losses.mean_squared_error(tf.cast(labels[1], tf.float32), preds[1])

        total_loss = ctr_loss * 0.5 + overall_loss

        #metrics ops
        ctr_auc = tf.metrics.auc(labels=labels[0], predictions=preds[0])
        ctr_eval_loss = tf.keras.metrics.BinaryCrossentropy()
        ctr_eval_loss.update_state(labels[0], preds[0])
        overall_eval_loss = tf.metrics.mean_squared_error(tf.cast(labels[1], tf.float32), preds[1])

        metric_dict = {
            "ctr_auc":ctr_auc,
            "ctr_loss":ctr_eval_loss,
            "overall_loss":overall_eval_loss
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=total_loss, eval_metric_ops=metric_dict)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=total_loss, train_op=train_op)


def ESMM(features, labels, mode, params):
    feature_inputs = {}
    # DIN model features
    for key, value in params["feature_configs"].used_features.items():
        if key not in ["label", "overall"]:
            feature_inputs[key] = fc.input_layer(features, value)

    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        item_embedding = tf.get_variable('item_embedding', [FLAGS.item_size, FLAGS.item_embedding_size])
        cate_embedding = tf.get_variable("cate_embedding", [FLAGS.cate_size, FLAGS.cate_embedding_size])
        inputs_seq_item_embedded = tf.nn.embedding_lookup(item_embedding,
                                                          tf.cast(feature_inputs["item_list"], dtype=tf.int64))
        inputs_seq_cate_embedded = tf.nn.embedding_lookup(cate_embedding,
                                                          tf.cast(feature_inputs["cate_list"], dtype=tf.int64))
        inputs_item_embedded = tf.nn.embedding_lookup(item_embedding,
                                                      tf.cast(feature_inputs["item_id"], dtype=tf.int64))
        inputs_cate_embedded = tf.nn.embedding_lookup(cate_embedding,
                                                      tf.cast(feature_inputs["cate_id"], dtype=tf.int64))

    # 加入用户历史评价作为权重
    # print("----------", tf.expand_dims(feature_inputs["overall_list"], axis=2).shape)
    inputs_seq_item_embedded_with_weight = inputs_seq_item_embedded * tf.expand_dims(feature_inputs["overall_list"],
                                                                                     axis=2)
    inputs_seq_cate_embedded_with_weight = inputs_seq_cate_embedded * tf.expand_dims(feature_inputs["overall_list"],
                                                                                     axis=2)

    seq_embedded = tf.concat([inputs_seq_item_embedded_with_weight, inputs_seq_cate_embedded_with_weight], axis=-1)
    target_embedded = tf.concat([inputs_item_embedded, inputs_cate_embedded], axis=-1)
    # target_embedded = tf.reshape(target_embedded, shape=(-1,target_embedded.shape[2]))
    # print("xxxxxxxx ",seq_embedded.shape, target_embedded.shape, inputs_item_embedded.shape, feature_inputs["cate_id"].shape)
    with tf.variable_scope("attention"):
        masks = tf.expand_dims(feature_inputs["item_list"] >= 0, axis=1)
        target_embedded = tf.reshape(target_embedded, shape=(-1, target_embedded.shape[2]))
        att_emb = attention(seq_embedded, target_embedded, masks, params)
    with tf.variable_scope("concat"):
        net = tf.layers.flatten(tf.concat([att_emb, target_embedded], axis=1))

    with tf.variable_scope("ctr"):
        for units in params['hidden_units']:
            ctr_model = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
                ctr_model = tf.layers.dropout(ctr_model, params['dropout_rate'],
                                               training=(mode == tf.estimator.ModeKeys.TRAIN))
        ctr_model = tf.layers.dense(ctr_model, units=1, name="ctr_output_layer")
    with tf.variable_scope("overall"):
        for units in params['hidden_units']:
            overall_model = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
                overall_model = tf.layers.dropout(overall_model, params['dropout_rate'],
                                               training=(mode == tf.estimator.ModeKeys.TRAIN))
        overall_model = tf.layers.dense(overall_model, units=1, name="ctr_output_layer")

    logits = [ctr_model, overall_model]

    preds = [tf.sigmoid(logits[0]), logits[1]]

    predictions = {
        "label": preds[0],
        "overall": preds[1]
    }

    labels = [labels["label"], labels["overall"]] if labels else None

    export_output = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)
    }

    # print("YYYYYYYYYYYYYYYYYYYYYYY",logits[0].shape, labels[0].shape)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_output)
    else:
        ctr_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[0], labels=tf.cast(labels[0], tf.float32)))
        overall_loss = tf.losses.mean_squared_error(tf.cast(labels[1], tf.float32), preds[1])

        total_loss = ctr_loss * 0.5 + overall_loss

        #metrics ops
        ctr_auc = tf.metrics.auc(labels=labels[0], predictions=preds[0])
        ctr_eval_loss = tf.keras.metrics.BinaryCrossentropy()
        ctr_eval_loss.update_state(labels[0], preds[0])
        overall_eval_loss = tf.metrics.mean_squared_error(tf.cast(labels[1], tf.float32), preds[1])

        metric_dict = {
            "ctr_auc":ctr_auc,
            "ctr_loss":ctr_eval_loss,
            "overall_loss":overall_eval_loss
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=total_loss, eval_metric_ops=metric_dict)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=total_loss, train_op=train_op)

def cgc(inputs, level_name, params, is_last):
    # inputs 形式 [task1, task2... shared task]
    # 单任务expert网络
    specific_expert_outputs = []
    for i in range(2):
        for j in range(params["expert_net_num"]):
            specific_expert_outputs.append(tf.expand_dims(
                tf.layers.dense(inputs[i], params["units"], activation="relu", use_bias=True, name="level{}_task{}_specific_expert{}".format(level_name, i, j)), axis=2
            ))
    #shared expert 网络
    shared_expert_outputs = []
    for j in range(params["expert_net_num"]):
        shared_expert_outputs.append(tf.expand_dims(
            tf.layers.dense(inputs[-1], params["units"], activation="relu", use_bias=True,
                            name="level{}_shared_expert{}".format(level_name, j)), axis=2
        ))
    #gate网络
    gate_outputs = []
    cgc_outputs = []
    for i in range(2):
        task_specific_expert_output = tf.concat(specific_expert_outputs[i*params["expert_net_num"]: (i+1)*params["expert_net_num"]] + shared_expert_outputs, axis=2)
        gate_output = tf.layers.dense(
            inputs[i], params["expert_net_num"]*2, activation="softmax", use_bias=True, name="level{}_task{}_specific_gate".format(level_name, i)
        )
        gate_outputs.append(gate_output)
        expand_gate_out = tf.expand_dims(gate_output, axis=1)
        weighted_expert_output = task_specific_expert_output * repeat_elements(expand_gate_out, params["units"], axis=1)
        cgc_outputs.append(sum(weighted_expert_output, axis=2))
    if not is_last:
        shared_expert_output = tf.concat(specific_expert_outputs + shared_expert_outputs, axis=2)
        gate_output = tf.layers.dense(
            inputs[-1], params["expert_net_num"]*3, activation="softmax", use_bias=True,
            name="level{}_task{}_shared_gate".format(level_name, i)
        )
        gate_outputs.append(gate_output)
        expand_gate_out = tf.expand_dims(gate_output, axis=1)
        weighted_expert_output = shared_expert_output * repeat_elements(expand_gate_out, params["units"], axis=1)
        cgc_outputs.append(sum(weighted_expert_output, axis=2))
    return cgc_outputs

def PLE(features, labels, mode, params):
    feature_inputs = {}
    # DIN model features
    for key, value in params["feature_configs"].used_features.items():
        if key not in ["label", "overall"]:
            feature_inputs[key] = fc.input_layer(features, value)

    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        item_embedding = tf.get_variable('item_embedding', [FLAGS.item_size, FLAGS.item_embedding_size])
        cate_embedding = tf.get_variable("cate_embedding", [FLAGS.cate_size, FLAGS.cate_embedding_size])
        inputs_seq_item_embedded = tf.nn.embedding_lookup(item_embedding,
                                                          tf.cast(feature_inputs["item_list"], dtype=tf.int64))
        inputs_seq_cate_embedded = tf.nn.embedding_lookup(cate_embedding,
                                                          tf.cast(feature_inputs["cate_list"], dtype=tf.int64))
        inputs_item_embedded = tf.nn.embedding_lookup(item_embedding,
                                                      tf.cast(feature_inputs["item_id"], dtype=tf.int64))
        inputs_cate_embedded = tf.nn.embedding_lookup(cate_embedding,
                                                      tf.cast(feature_inputs["cate_id"], dtype=tf.int64))

    # 加入用户历史评价作为权重
    # print("----------", tf.expand_dims(feature_inputs["overall_list"], axis=2).shape)
    inputs_seq_item_embedded_with_weight = inputs_seq_item_embedded * tf.expand_dims(feature_inputs["overall_list"],
                                                                                     axis=2)
    inputs_seq_cate_embedded_with_weight = inputs_seq_cate_embedded * tf.expand_dims(feature_inputs["overall_list"],
                                                                                     axis=2)

    seq_embedded = tf.concat([inputs_seq_item_embedded_with_weight, inputs_seq_cate_embedded_with_weight], axis=-1)
    target_embedded = tf.concat([inputs_item_embedded, inputs_cate_embedded], axis=-1)
    # target_embedded = tf.reshape(target_embedded, shape=(-1,target_embedded.shape[2]))
    # print("xxxxxxxx ",seq_embedded.shape, target_embedded.shape, inputs_item_embedded.shape, feature_inputs["cate_id"].shape)
    with tf.variable_scope("attention"):
        masks = tf.expand_dims(feature_inputs["item_list"] >= 0, axis=1)
        target_embedded = tf.reshape(target_embedded, shape=(-1, target_embedded.shape[2]))
        att_emb = attention(seq_embedded, target_embedded, masks, params)
    with tf.variable_scope("concat"):
        net = tf.layers.flatten(tf.concat([att_emb, target_embedded], axis=1))

    net = [net] * 3
    logits = []
    with tf.variable_scope("PLE"):
        #cgc levels
        for i in range(params["ple_levels"]):
            if i == params["ple_levels"]-1:
                net = cgc(net, i, params, True)
            else:
                net = cgc(net, i, params, False)
        #gate网络后的双塔部分
        for index, task_layer in enumerate(net):
            tower_layer = tf.layers.dense(task_layer, units=8, activation="relu")
            output_layer = tf.layers.dense(tower_layer, units=1, activation=None)
            # logits.append(tf.squeeze(output_layer, axis=1))
            logits.append(output_layer)

    # logits = [ctr_model, overall_model]

    preds = [tf.sigmoid(logits[0]), logits[1]]

    predictions = {
        "label": preds[0],
        "overall": preds[1]
    }

    labels = [labels["label"], labels["overall"]] if labels else None

    export_output = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)
    }

    # print("YYYYYYYYYYYYYYYYYYYYYYY",logits[0].shape, labels[0].shape)
    if mode == tf.estimator.ModeKeys.PREDICT:
        # return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_output=export_output)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_output)
    else:
        ctr_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[0], labels=tf.cast(labels[0], tf.float32)))
        overall_loss = tf.losses.mean_squared_error(tf.cast(labels[1], tf.float32), preds[1])

        total_loss = ctr_loss * 0.5 + overall_loss

        # metrics ops
        ctr_auc = tf.metrics.auc(labels=labels[0], predictions=preds[0])
        ctr_eval_loss = tf.keras.metrics.BinaryCrossentropy()
        ctr_eval_loss.update_state(labels[0], preds[0])
        overall_eval_loss = tf.metrics.mean_squared_error(tf.cast(labels[1], tf.float32), preds[1])

        metric_dict = {
            "ctr_auc": ctr_auc,
            "ctr_loss": ctr_eval_loss,
            "overall_loss": overall_eval_loss
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=total_loss,
                                              eval_metric_ops=metric_dict)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=total_loss, train_op=train_op)
