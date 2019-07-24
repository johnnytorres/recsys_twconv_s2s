

def create_tfrecord_pickle():
	my_dict = {'features' : {
	    'my_ints': [5, 6],
	    'my_float': [2.7],
	    'my_bytes': ['data']
	}}
	
	
	import pickle
	
	my_dict_str = pickle.dumps(my_dict)
	with open('my_dict.pkl', 'wb') as f:
	    f.write(my_dict_str)
	
	with open('my_dict.pkl', 'rb') as f:
	    that_dict_str = f.read()
	that_dict = pickle.loads(that_dict_str)


def create_tfrecord():
	import tensorflow as tf
	tf.enable_eager_execution()
	
	my_example = tf.train.Example(features=tf.train.Features(feature={
	    'my_ints': tf.train.Feature(int64_list=tf.train.Int64List(value=[5, 6])),
	    'my_float': tf.train.Feature(float_list=tf.train.FloatList(value=[2.7])),
	    'my_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes('data', 'utf-8')]))
	}))
	
	my_example_str = my_example.SerializeToString()
	with tf.python_io.TFRecordWriter('my_example.tfrecords') as writer:
	    writer.write(my_example_str)
		
	


def read_tfrecord():
	import tensorflow as tf
	tf.enable_eager_execution()
	
	reader = tf.python_io.tf_record_iterator('my_example.tfrecords')

	# feature_spec = {
	# 	'my_ints': tf.FixedLenFeature(shape=[], dtype=tf.int64),
	# 	'my_float': tf.FixedLenFeature(shape=[], dtype=tf.float32),
	# 	'my_bytes': tf.FixedLenFeature(shape=[], dtype=tf.string),
	# }
	
	feature_spec = {
		'my_ints': tf.VarLenFeature(dtype=tf.int64),
		'my_float': tf.VarLenFeature( dtype=tf.float32),
		'my_bytes': tf.VarLenFeature( dtype=tf.string),
	}
	
	ser = next(reader)
	e = tf.train.Example().FromString(ser)
	tfe = tf.parse_single_example(serialized=ser, features=feature_spec)
	
	those_examples = []
	
	print(those_examples)


def read_train_file():
	import tensorflow as tf
	tf.enable_eager_execution()
	
	reader = tf.python_io.tf_record_iterator('/users/johnny/data/ubuntu/ubuntu_small/example.tfrecords')
	
	# feature_spec = {
	# 	'my_ints': tf.FixedLenFeature(shape=[], dtype=tf.int64),
	# 	'my_float': tf.FixedLenFeature(shape=[], dtype=tf.float32),
	# 	'my_bytes': tf.FixedLenFeature(shape=[], dtype=tf.string),
	# }
	
	feature_spec = {
		'context': tf.FixedLenFeature(shape=160, dtype=tf.int64),
		'utterance': tf.FixedLenFeature(shape=160, dtype=tf.int64),
		'label': tf.FixedLenFeature(shape=1,dtype=tf.int64),
	}
	
	for ser in reader:
		e = tf.train.Example().FromString(ser)
		#tfe = tf.parse_single_example(serialized=ser, features=feature_spec)
		print(e)
	
	#those_examples = []
	#print(those_examples)
	
#read_tfrecord()
read_train_file()
