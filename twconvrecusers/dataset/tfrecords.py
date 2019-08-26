import os
import csv
import functools
import tensorflow as tf
import logging
import subprocess
from tqdm import tqdm


tf.flags.DEFINE_string(
	name="input_dir", 
	default="~/dataset",
	help="Input directory contakining original CSV dataset files"
	)
tf.flags.DEFINE_integer(
	name="min_word_frequency", 
	default=5, 
	help="Minimum frequency of words in the vocabulary"
	)
tf.flags.DEFINE_integer(
	name="max_sentence_len", 
	default=160, 
	help="Maximum Sentence Length"
	)
tf.flags.DEFINE_integer(
	name="num_distractors", 
	default=9,
	help="Number of distractor"
	)	
tf.flags.DEFINE_boolean(
	name='example', 
	default=False, 
	help='indicates if create only a sample file for prediction'
	)

FLAGS = tf.flags.FLAGS

train_prefix = 'train'
valid_prefix = 'valid'
test_prefix = 'test'

input_dir=os.path.expanduser(FLAGS.input_dir)
TRAIN_PATH = os.path.join(input_dir, f"{train_prefix}.csv")
VALID_PATH = os.path.join(input_dir, f"{valid_prefix}.csv")
TEST_PATH = os.path.join(input_dir, f"{test_prefix}.csv")
FLAGS.input_dir=input_dir


def tokenizer_fn(iterator):
	return (x.split(" ") for x in iterator)


def textfields_csv_iter(trainfile, testfile=None, validfile=None):
	"""
	Returns an iterator over a CSV file. Skips the header.
	"""
	# define which fields to include
	files_fields = [
		range(2), # for training (context, profile) 
		range(2 + FLAGS.num_distractors), # for validation and testing (context, profile , distractors...)
		range(2 + FLAGS.num_distractors), # for validation and testing (context, profile , distractors...)
		]
	files_paths = [
		trainfile, 
		testfile, 
		validfile
		]

	for path, fields in zip(files_paths, files_fields):
		if path is None:
			continue
		num_lines = wccount(path)
		with open(path) as csvfile:
			reader = csv.reader(csvfile)
			# Skip the header
			next(reader)
			for row in tqdm(reader, f'{os.path.split(path)[1]}', total=num_lines):
				yield ' '.join([row[i] for i in fields])


def fields_csv_iter(trainfile):
	path = trainfile
	num_lines = wccount(path)
	with open(path) as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for row in tqdm(reader, f'{os.path.split(path)[1]}', total=num_lines):
			yield row


def create_vocab(input_iter, min_frequency):
	"""
	Creates and returns a VocabularyProcessor object with the vocabulary
	for the input iterator.
	"""
	vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
		FLAGS.max_sentence_len,
		min_frequency=min_frequency,
		tokenizer_fn=tokenizer_fn)
	vocab_processor.fit(input_iter)
	return vocab_processor


def transform_sentence(sequence, vocab_processor):
	"""
	Maps a single sentence into the integer vocabulary. Returns a python array.
	"""
	return next(vocab_processor.transform([sequence])).tolist()


def create_example_train(row, vocab):
	"""
	Creates a training example for the Ubuntu Dialog Corpus dataset.
	Returnsthe a tensorflow.Example Protocol Buffer object.
	"""
	context, utterance, label = row
	context_transformed = transform_sentence(context, vocab)
	utterance_transformed = transform_sentence(utterance, vocab)
	context_len = len(next(vocab._tokenizer([context])))
	utterance_len = len(next(vocab._tokenizer([utterance])))
	label = int(float(label))
	
	# New Example
	example = tf.train.Example()
	example.features.feature["context"].int64_list.value.extend(context_transformed)
	example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
	example.features.feature["context_len"].int64_list.value.extend([context_len])
	example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
	example.features.feature["label"].int64_list.value.extend([label])
	return example


def create_example_test(row, vocab):
	"""
	Creates a tests/validation example for the Ubuntu Dialog Corpus dataset.
	Returnsthe a tensorflow.Example Protocol Buffer object.
	"""
	context, utterance = row[:2]
	distractors = row[2:]
	context_len = len(next(vocab._tokenizer([context])))
	utterance_len = len(next(vocab._tokenizer([utterance])))
	context_transformed = transform_sentence(context, vocab)
	utterance_transformed = transform_sentence(utterance, vocab)
	
	# New Example
	example = tf.train.Example()
	example.features.feature["context"].int64_list.value.extend(context_transformed)
	example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
	example.features.feature["context_len"].int64_list.value.extend([context_len])
	example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
	
	# Distractor sequences
	for i, distractor in enumerate(distractors):
		dis_key = "distractor_{}".format(i)
		dis_len_key = "distractor_{}_len".format(i)
		# Distractor Length Feature
		dis_len = len(next(vocab._tokenizer([distractor])))
		example.features.feature[dis_len_key].int64_list.value.extend([dis_len])
		# Distractor Text Feature
		dis_transformed = transform_sentence(distractor, vocab)
		example.features.feature[dis_key].int64_list.value.extend(dis_transformed)
	return example


def wccount(filename):
	out = subprocess.Popen(['wc', '-l', filename],
	                       stdout=subprocess.PIPE,
	                       stderr=subprocess.STDOUT
	                       ).communicate()[0]
	print(out)
	return int(out.split()[0])


def create_tfrecords_file(input_filename, output_filename, example_fn):
	"""
	Creates a TFRecords file for the given input dataset and
	example transofmration function
	"""
	writer = tf.io.TFRecordWriter(output_filename)
	num_examples = wccount(input_filename)
	
	for i, row in enumerate(fields_csv_iter(input_filename)):
		x = example_fn(row)
		writer.write(x.SerializeToString())

	writer.close()


def write_vocabulary(vocab_processor, outfile):
	"""
	Writes the vocabulary to a file, one word per line.
	"""
	vocab_size = len(vocab_processor.vocabulary_)
	with open(outfile, "w") as vocabfile:
		for id in range(vocab_size):
			word = vocab_processor.vocabulary_._reverse_mapping[id]
			vocabfile.write(word + "\n")
	logging.info("Saved vocabulary to {}".format(outfile))


def create_example():
	vocab = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
		os.path.join(FLAGS.output_dir, "vocab_processor.bin"))
	
	TRAIN_PATH = os.path.join(os.path.expanduser(FLAGS.input_dir), f"examples.csv")
	
	create_tfrecords_file(
		input_filename=TRAIN_PATH,
		output_filename=os.path.join(FLAGS.output_dir, f"example.tfrecords"),
		example_fn=functools.partial(create_example_train, vocab=vocab))


def create_tfrecords():
	print("Creating vocabulary...")
	input_iter = textfields_csv_iter(TRAIN_PATH, VALID_PATH, TEST_PATH)
	vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
	vocab_size = len(vocab.vocabulary_)
	print("Total vocabulary size: {}".format(vocab_size))
	with open(os.path.join(FLAGS.input_dir, 'vocab_size.txt'), 'w') as f:
		f.write(str(vocab_size)+'\n')
	# Create vocabulary.txt file
	write_vocabulary(
		vocab, os.path.join(FLAGS.input_dir, "vocabulary.txt"))
	# Save vocab processor
	vocab.save(os.path.join(FLAGS.input_dir, "vocab_processor.bin"))
	# Create validation.tfrecords
	print('Creating tfrecords...')
	create_tfrecords_file(
		input_filename=VALID_PATH,
		output_filename=os.path.join(FLAGS.input_dir, f"{valid_prefix}.tfrecords"),
		example_fn=functools.partial(create_example_test, vocab=vocab))
	# Create tests.tfrecords
	create_tfrecords_file(
		input_filename=TEST_PATH,
		output_filename=os.path.join(FLAGS.input_dir, f"{test_prefix}.tfrecords"),
		example_fn=functools.partial(create_example_test, vocab=vocab))
	# Create train.tfrecords
	create_tfrecords_file(
		input_filename=TRAIN_PATH,
		output_filename=os.path.join(FLAGS.input_dir, f"{train_prefix}.tfrecords"),
		example_fn=functools.partial(create_example_train, vocab=vocab))

	if FLAGS.example:
		create_example()

	print('done')


if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

	create_tfrecords()
