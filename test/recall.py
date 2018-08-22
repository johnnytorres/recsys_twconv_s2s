

import tensorflow as tf


#tf.enable_eager_execution()

with tf.Session() as sess:
	labels = tf.constant(value=[[0],[0]], dtype=tf.int64)
	probs = tf.constant(value=[[.1, 1, 0.9,],[1, 1, 1]])
	c=tf.metrics.recall_at_k(labels, probs, k=1)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	print(labels)
	print(sess.run(c))
#print()