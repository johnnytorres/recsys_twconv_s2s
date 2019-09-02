
import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow.python.ops.losses import losses
from twconvrecusers.models.neural import build_embedding_layer, update_learning_rate, get_feature, metric_fn


def create_estimator(config, HYPER_PARAMS):

    def _matrix_factorization(context, utterance, context_len, utterance_len):
        """ Create the model structure and compute the logits """
        embeddings_layer = build_embedding_layer(HYPER_PARAMS)
        # embed the context and utterances
        context_embedded = tf.nn.embedding_lookup(
            embeddings_layer, context, name='embed_context'
        )
        utterance_embedded = tf.nn.embedding_lookup(
            embeddings_layer, utterance, name='embed_utterance'
        )

        with tf.variable_scope('F') as vs:
            # dot product between the generated utterance and the actual utterance
            context_embedded = tf.contrib.layers.flatten(context_embedded)
            context_embedded = tf.expand_dims(context_embedded, -1)
            utterance_embedded = tf.contrib.layers.flatten(utterance_embedded)
            utterance_embedded = tf.expand_dims(utterance_embedded, -1)
            logits = tf.matmul(context_embedded, utterance_embedded, transpose_a=True)
            logits = tf.squeeze(logits, [2], name='conv_logits')
            return logits

    def _inference(features):
        """ compute the logits """
        # context=features['context']
        # utterance=features['utterance']
        context, contex_len = get_feature(
            features, 'context', 'context_len',
            HYPER_PARAMS.max_content_len)
        utterance, utterance_len = get_feature(
            features, 'utterance', 'utterance_len',
            HYPER_PARAMS.max_utterance_len
        )

        return _matrix_factorization(context, utterance, contex_len, utterance_len)

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""

        # Update learning rate using exponential decay method
        current_learning_rate = update_learning_rate(HYPER_PARAMS)

        # Create Optimiser
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=current_learning_rate
        )

        # Create training operation
        #loss += tf.losses.get_regularization_loss()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),

        )

        return train_op

    def _model_fn(features, labels, mode):
        """ model function for the custom estimator"""
        logits = _inference(features)

        if HYPER_PARAMS.debug:
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

    tf.compat.v1.logging.info("creating a custom model...")

    estimator = tf.estimator.Estimator(model_fn=_model_fn, config=config)

    estimator = tf.contrib.estimator.add_metrics(
        estimator,
        lambda labels, predictions: metric_fn(HYPER_PARAMS, labels, predictions) )

    return estimator

