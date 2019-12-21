
import os
import array
import numpy as np
import tensorflow as tf
from collections import defaultdict


def load_vocab(filename):
	vocab_size_path = os.path.join(os.path.split(filename)[0], 'vocab_size.txt')
	with tf.io.gfile.GFile(vocab_size_path) as f:
		vocab_size = int(f.read())
	with tf.gfile.GFile(filename) as f:
		vocab = f.read().splitlines()
	dct = defaultdict(int)
	tf.compat.v1.logging.info('loading vocabulary...')
	for idx, word in enumerate(vocab):
		dct[word] = idx
	tf.compat.v1.logging.info('loading vocabulary... found {} words'.format(len(dct)))
	return [vocab, dct, vocab_size]


def load_embedding_vectors(embedding_path, vocab_dict, small_embedding_path=None):
	"""
	Load embedding vectors from a .txt file.
	Optionally limit the vocabulary to save memory. `vocab` should be a set.
	"""
	dct = {}
	vectors = array.array('d')
	current_idx = 0
	
	small_embeddings = []
	
	with tf.gfile.GFile(embedding_path) as f:
		num_embeddings, embeddings_dim  = next(f).split(' ')
		num_embeddings=int(num_embeddings)
		tf.compat.v1.logging.info('loading embeddings...')
		for _, line in enumerate(f):
			tokens = line.rstrip().split(" ")
			word = tokens[0]
			entries = tokens[1:]
			if word in vocab_dict:
				dct[word] = current_idx
				vectors.extend(float(x) for x in entries)
				current_idx += 1
				small_embeddings.append(line)
					
	word_dim = len(entries)
	num_vectors = len(dct)
	tf.compat.v1.logging.info("Found embeddings for {} out of {} words in vocabulary".format(num_vectors, len(vocab_dict)))
	
	if small_embedding_path:
		with open(small_embedding_path, 'w') as f:
			f.writelines(['{} {}\n'.format(num_vectors,word_dim)])
			f.writelines(small_embeddings)
			
	return [np.array(vectors).reshape(num_vectors, word_dim), dct]




