

import tensorflow as tf


x = tf.constant([[2.0, 2.0], [2.0, 2.0]])
w = tf.Variable([[2, 1], [2, 1]], dtype=tf.float32)
#w = tf.expand_dims(w, 1)
print(x,w)
y = tf.matmul(x,w)
print(y)
output = tf.nn.softmax(y, 1)
print(output)

with tf.Session() as sess:
	sess.run(w.initializer)
	print(sess.run(w))
	print(sess.run(y))
	print(sess.run(output))
