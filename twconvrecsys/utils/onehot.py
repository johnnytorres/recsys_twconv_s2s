


import tensorflow as tf


tf.enable_eager_execution()

#with tf.Session() as sess:

labels = tf.constant(value=[[0],[5]], dtype=tf.int64)

labels_onehot = tf.one_hot(labels, depth=6)

target_lbl = tf.gather(labels_onehot, 0, axis=2)

decoded = tf.argmax(labels_onehot, axis=2)

print(labels.eval())

