

import tensorflow as tf

tf.enable_eager_execution()

batch = tf.constant(value=[[0,1,5,6,7],[5,3,2,1,5]], dtype=tf.int64)
all_sources=[batch, batch]

r = tf.concat(all_sources,0)

print('done')
