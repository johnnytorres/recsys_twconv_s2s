

import tensorflow as tf
import tensorflow.contrib as tfc


#tf.enable_eager_execution()

with tf.Session() as sess:
	labels = tf.constant(value=[[1],[0],[2]], dtype=tf.int64)
	probs = tf.constant(value=[[0.8, 0.93, .2,.1],[.82, 0, .1,.83],[.92,.1, .50, .3]])
	c = tf.metrics.recall_at_k(predictions= probs, labels= labels, k=2)
	d = tfc.metrics.streaming_sparse_recall_at_k(predictions=probs, labels=labels, k=2)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	print(sess.run(c))
	print(sess.run(d))
	print('done')