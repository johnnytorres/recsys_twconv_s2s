
import array
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow.python.ops.losses import losses


from twconvrecusers.dataset import embeddings, featurizer, metadata

MODEL_RNN = 'rnn'
MODEL_LSTM = 'lstm'
MODEL_BiLSTM = 'bilstm'

EMBEDDING_LAYER_NAME = 'word_embeddings'
vocab_array = None
vocab_dict = None
vocab_size = None
embedding_vectors = None
embbeding_dict = None




def construct_hidden_units():
    """ Create the number of hidden units in each layer

	if the HYPER_PARAMS.layer_sizes_scale_factor > 0 then it will use a "decay" mechanism
	to define the number of units in each layer. Otherwise, task.HYPER_PARAMS.hidden_units
	will be used as-is.

	Returns:
		list of int
	"""
    hidden_units = list(map(int, task.HYPER_PARAMS.hidden_units.split(',')))

    if task.HYPER_PARAMS.layer_sizes_scale_factor > 0:
        first_layer_size = hidden_units[0]
        scale_factor = task.HYPER_PARAMS.layer_sizes_scale_factor
        num_layers = task.HYPER_PARAMS.num_layers

        hidden_units = [
            max(2, int(first_layer_size * scale_factor ** i))
            for i in range(num_layers)
        ]

    tf.logging.info(("Hidden units structure: {}".format(hidden_units)))

    return hidden_units


def update_learning_rate(HYPER_PARAMS):
    """ Updates learning rate using an exponential decay method

	Returns:
	   float - updated (decayed) learning rate
	"""
    initial_learning_rate = HYPER_PARAMS.learning_rate
    decay_steps = HYPER_PARAMS.train_steps  # decay after each training step
    decay_factor = HYPER_PARAMS.learning_rate_decay_factor  # if set to 1, then no decay.

    global_step = tf.train.get_global_step()

    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_factor)

    return learning_rate


def parse_label_column(label_string_tensor):
    """ Convert string class labels to indices

	Returns:
	   Tensor of type int
	"""
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(metadata.TARGET_LABELS))
    return table.lookup(label_string_tensor)


def create_classifier(config):
    """ Create a DNNLinearCombinedClassifier based on the HYPER_PARAMS in task.py

	Args:
		config - used for model directory
	Returns:
		DNNLinearCombinedClassifier
	"""

    feature_columns = list(featurizer.create_feature_columns().values())

    deep_columns, wide_columns = featurizer.get_deep_and_wide_columns(
        feature_columns
    )

    # Change the optimisers for the wide and deep parts of the model if you wish
    linear_optimizer = tf.train.FtrlOptimizer(learning_rate=task.HYPER_PARAMS.learning_rate)
    dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=task.HYPER_PARAMS.learning_rate)

    estimator = tf.estimator.DNNLinearCombinedClassifier(

        n_classes=len(metadata.TARGET_LABELS),
        label_vocabulary=metadata.TARGET_LABELS,

        linear_optimizer=linear_optimizer,
        linear_feature_columns=wide_columns,

        dnn_feature_columns=deep_columns,
        dnn_optimizer=dnn_optimizer,

        weight_column=metadata.WEIGHT_COLUMN_NAME,

        dnn_hidden_units=construct_hidden_units(),
        dnn_activation_fn=tf.nn.relu,
        dnn_dropout=task.HYPER_PARAMS.dropout_prob,

        config=config,
    )

    estimator = tf.contrib.estimator.add_metrics(estimator, metric_fn)

    tf.logging.info(("creating a classification model: {}".format(estimator)))

    return estimator


def create_regressor(config):
    """ Create a DNNLinearCombinedRegressor based on the HYPER_PARAMS in task.py

	Args:
		config - used for model directory
	Returns:
		DNNLinearCombinedRegressor
	"""

    feature_columns = list(featurizer.create_feature_columns().values())

    deep_columns, wide_columns = featurizer.get_deep_and_wide_columns(
        feature_columns
    )

    # Change the optimisers for the wide and deep parts of the model if you wish
    linear_optimizer = tf.train.FtrlOptimizer(learning_rate=task.HYPER_PARAMS.learning_rate)
    dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=task.HYPER_PARAMS.learning_rate)

    estimator = tf.estimator.DNNLinearCombinedRegressor(

        linear_optimizer=linear_optimizer,
        linear_feature_columns=wide_columns,

        dnn_feature_columns=deep_columns,
        dnn_optimizer=dnn_optimizer,

        weight_column=metadata.WEIGHT_COLUMN_NAME,

        dnn_hidden_units=construct_hidden_units(),
        dnn_activation_fn=tf.nn.relu,
        dnn_dropout=task.HYPER_PARAMS.dropout_prob,

        config=config,
    )

    estimator = tf.contrib.estimator.add_metrics(estimator, metric_fn)

    tf.logging.info(("creating a regression model: {}".format(estimator)))

    return estimator


def create_estimator(config, HYPER_PARAMS):

    def load_vocab(filename):
        vocab = None
        with tf.gfile.GFile(filename) as f:
            vocab = f.read().splitlines()
        dct = defaultdict(int)
        for idx, word in enumerate(vocab):
            dct[word] = idx
        return [vocab, dct]

    def load_embedding_vectors(filename, vocab):
        """
		Load embedding vectors from a .txt file.
		Optionally limit the vocabulary to save memory. `vocab` should be a set.
		"""
        dct = {}
        vectors = array.array('d')
        current_idx = 0
        with tf.gfile.GFile(filename)as f:  # , encoding="utf-8") as f:
            for _, line in enumerate(f):
                tokens = line.split(" ")
                word = tokens[0]
                entries = tokens[1:]
                if not vocab or word in vocab:
                    dct[word] = current_idx
                    vectors.extend(float(x) for x in entries)
                    current_idx += 1
            word_dim = len(entries)
            num_vectors = len(dct)
            tf.logging.info("Found {} words out of {} in embeddings.".format(num_vectors, len(vocab)))
            return [np.array(vectors).reshape(num_vectors, word_dim), dct]

    def build_initial_embedding_matrix(vocab_dict, embedding_dict, embedding_vectors, embedding_dim):
        initial_embeddings = np.random.uniform(-0.25, 0.25, (len(vocab_dict), embedding_dim)).astype("float32")
        for word, glove_word_idx in embedding_dict.items():
            word_idx = vocab_dict.get(word)
            initial_embeddings[word_idx, :] = embedding_vectors[glove_word_idx]
        return initial_embeddings

    def build_embedding_layer():
        global vocab_array, vocab_dict, vocab_size, embedding_vectors, embbeding_dict

        if vocab_array is None:
            vocab_array, vocab_dict, vocab_size = embeddings.load_vocab(HYPER_PARAMS.vocab_path)

        if HYPER_PARAMS.embedding_path:
            tf.logging.info('loading embeddings...')
            if embedding_vectors is None:
                embedding_vectors, embbeding_dict = embeddings.load_embedding_vectors(
                    HYPER_PARAMS.embedding_path, set(vocab_array))
            initializer = build_initial_embedding_matrix(vocab_dict, embbeding_dict, embedding_vectors,
                                                         HYPER_PARAMS.embedding_size)
            embedding_layer = tf.get_variable(
                name=EMBEDDING_LAYER_NAME,
                initializer=initializer,
                trainable=HYPER_PARAMS.embedding_trainable
            )
        else:
            tf.logging.info('No embeddings specified, starting with random embeddings!')
            initializer = tf.random_normal_initializer(-0.25, 0.25)  # todo: maybe hyperparam?
            #initializer = tf.glorot_uniform_initializer()

            if HYPER_PARAMS.vocab_size == -1:
                HYPER_PARAMS.vocab_size = vocab_size

            embedding_layer = tf.get_variable(
                name=EMBEDDING_LAYER_NAME,
                shape=[HYPER_PARAMS.vocab_size, HYPER_PARAMS.embedding_size],
                initializer=initializer
            )

        return embedding_layer

    def _get_feature(features, key, key_len, max_len):
        ids = features[key]
        # ids = tf.squeeze(features[key], [1])
        ids_len = tf.squeeze(features[key_len], [1])
        ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
        return ids, ids_len

    def _dual_encoder(context, utterance, context_len, utterance_len):
        """ Create the model structure and compute the logits """
        embeddings_layer = build_embedding_layer()
        # embed the context and utterances
        context_embedded = tf.nn.embedding_lookup(
            embeddings_layer, context, name='embed_context'
        )
        utterance_embedded = tf.nn.embedding_lookup(
            embeddings_layer, utterance, name='embed_utterance'
        )

        # create sequence layer based on LSTM
        with tf.variable_scope('rnn') as vs:

            if HYPER_PARAMS.estimator == MODEL_RNN:
                cell = tf.nn.rnn_cell.BasicRNNCell(
                    num_units=HYPER_PARAMS.rnn_dim,
                )
            elif HYPER_PARAMS.estimator == MODEL_LSTM:
                cell = tf.nn.rnn_cell.LSTMCell(
                    num_units=HYPER_PARAMS.rnn_dim,
                    initializer=tf.glorot_uniform_initializer(),
                    # todo: hyperparameters?
                    forget_bias=2.0,
                    use_peepholes=True,
                    state_is_tuple=True
                )
            else:
                cell_fw = tf.nn.rnn_cell.BasicLSTMCell(
                    num_units=HYPER_PARAMS.rnn_dim,
                    forget_bias=2.0,
                    state_is_tuple=True
                )
                cell_bw = tf.nn.rnn_cell.BasicLSTMCell(
                    num_units=HYPER_PARAMS.rnn_dim,
                    forget_bias=2.0,
                    state_is_tuple=True
                )

            if HYPER_PARAMS.estimator == MODEL_BiLSTM:
                rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=tf.concat([context_embedded, utterance_embedded], 0),
                    sequence_length=tf.concat([context_len, utterance_len], 0),
                    dtype=tf.float32
                )
            else:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=tf.concat([context_embedded, utterance_embedded], 0),
                    sequence_length=tf.concat([context_len, utterance_len], 0),
                    dtype=tf.float32
                )

            if HYPER_PARAMS.estimator == MODEL_RNN:
                context_encoded, utterance_encoded = tf.split(rnn_states, 2, 0)
            elif HYPER_PARAMS.estimator == MODEL_LSTM:
                context_encoded, utterance_encoded = tf.split(rnn_states.h, 2, 0)
            else:
                rnn_state_fw, rnn_state_bw = rnn_states
                # todo: other strategy is to concat, but M must match
                rnn_states = (rnn_state_fw.h + rnn_state_bw.h) / 2
                context_encoded, utterance_encoded = tf.split(rnn_states, 2, 0)

        with tf.variable_scope('F') as vs:
            M = tf.get_variable(
                'M', shape=[HYPER_PARAMS.rnn_dim,HYPER_PARAMS.rnn_dim],
                initializer=tf.truncated_normal_initializer(),
                #regularizer=tf.re
                #regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )

            # predict a user utterance ( c * M )
            utterances_generated = tf.matmul(context_encoded, M, name='utterances_generated')
            utterances_generated = tf.expand_dims(utterances_generated, 2)
            utterance_encoded = tf.expand_dims(utterance_encoded, 2)

            # dot product between the generated utterance and the actual utterance
            logits = tf.matmul(utterances_generated, utterance_encoded, transpose_a=True)
            logits = tf.squeeze(logits, [2], name='conv_logits')
            return logits

    def _inference(features):
        """ compute the logits """
        # context=features['context']
        # utterance=features['utterance']
        context, contex_len = _get_feature(
            features, 'context', 'context_len',
            HYPER_PARAMS.max_content_len)
        utterance, utterance_len = _get_feature(
            features, 'utterance', 'utterance_len',
            HYPER_PARAMS.max_utterance_len
        )

        return _dual_encoder(context, utterance, contex_len, utterance_len)

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""

        # Update learning rate using exponential decay method
        current_learning_rate = update_learning_rate(HYPER_PARAMS)

        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(
            learning_rate=current_learning_rate
        )

        # Create training operation
        #loss += tf.losses.get_regularization_loss()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),

        )

        return train_op

    def metric_fn(labels, predictions):
        """ Defines extra metrics metrics to canned and custom estimators.
    	By default, this returns an empty dictionary

    	Args:
    		labels: A Tensor of the same shape as predictions
    		predictions: A Tensor of arbitrary shape
    	Returns:
    		dictionary of string:metric
    	"""
        metrics = {}

        num_instances_recall = HYPER_PARAMS.num_distractors + 1
        # probs = tf.tf.logging.info(predictions, [predictions], 'calculating metric recall @k predictions')

        probs = predictions['logistic']

        split_predictions = tf.split(probs, num_instances_recall, 0)
        concat_predictions = tf.concat(split_predictions, 1)

        recall_labels = tf.zeros(shape=(tf.shape(concat_predictions)[0], 1), dtype=tf.int64, name='recall_labels')

        # concat_predictions = tf.Print(
        #     concat_predictions,
        #     [concat_predictions, recall_labels],
        #     'calculating metric recall @k split probs',
        #     summarize=10)

        # TODO: the k metrics depends of the number of distractors
        for k in [1, 2, 5]:  # , 10]:
            metric_name = "recall_at_%d" % k
            metrics[metric_name] = tf.metrics.recall_at_k(
                recall_labels,
                concat_predictions,
                k,
                name=metric_name
            )

        # Example of implementing Root Mean Squared Error for regression

        # pred_values = predictions['predictions']
        # metrics['rmse'] = tf.metrics.root_mean_squared_error(labels=labels,
        #                                                      predictions=pred_values)

        # Example of implementing Mean per Class Accuracy for classification

        # indices = parse_label_column(labels)
        # pred_class = predictions['class_ids']
        # metrics['mirco_accuracy'] = tf.metrics.mean_per_class_accuracy(labels=indices,
        #                                                                predictions=pred_class,
        #                                                                num_classes=len(metadata.TARGET_LABELS))

        return metrics

    def _model_fn(features, labels, mode):
        """ model function for the custom estimator"""
        logits = _inference(features)

        if mode == tfc.learn.ModeKeys.EVAL:
            logits = tf.Print(logits, [logits])

        head = tfc.estimator.binary_classification_head(
            loss_reduction=losses.Reduction.MEAN,
        )

        return head.create_estimator_spec(
            features,
            mode,
            logits,
            labels=labels,
            train_op_fn=_train_op_fn
        )

    tf.logging.info("creating a custom model...")

    estimator = tf.estimator.Estimator(model_fn=_model_fn, config=config)

    estimator = tf.contrib.estimator.add_metrics(estimator, metric_fn)

    return estimator

