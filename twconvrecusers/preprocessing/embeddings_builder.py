
import os
import csv
import argparse
import array
import numpy as np
import tensorflow as tf
from collections import defaultdict


def load_vocab(filename):
	vocab = None
	
	vocab_size_path = os.path.join(os.path.split(filename)[0], 'vocab_size.txt')
	with tf.gfile.GFile(vocab_size_path) as f:
		vocab_size = int(f.read())
	
	with tf.gfile.GFile(filename) as f:
		vocab = f.read().splitlines()
	dct = defaultdict(int)
	#for idx, word in tqdm(enumerate(vocab), 'loading vocabulary', total=vocab_size):
	for idx, word in enumerate(vocab):
		dct[word] = idx
	tf.logging.info('loading vocabulary... found {} words'.format(len(dct)))
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
		#for _, line in tqdm( enumerate(f), 'loading embeddings' , total=num_embeddings):
		for _, line in enumerate(f):
			tokens = line.rstrip().split(" ")
			word = tokens[0]
			entries = tokens[1:]
			if word in vocab_dict:
				dct[word] = current_idx
				vectors.extend(float(x) for x in entries)
				current_idx += 1
				#if small_embedding_path: # dont validate here to avoid performance issue
				small_embeddings.append(line)
					
	word_dim = len(entries)
	num_vectors = len(dct)
	tf.logging.info("Found embeddings for {} out of {} words in vocabulary".format(num_vectors, len(vocab_dict)))
	
	if small_embedding_path:
		with open(small_embedding_path, 'w') as f:
			f.writelines(['{} {}\n'.format(num_vectors,word_dim)])
			f.writelines(small_embeddings)
			
	return [np.array(vectors).reshape(num_vectors, word_dim), dct]


def run(args):
	tf.logging.set_verbosity(args.verbosity)
	vocab, vocab_dic, vocab_size = load_vocab(args.vocab_path)
	
	reduced_embeddings_path = os.path.join(os.path.split(args.vocab_path)[0], 'embeddings.vec')
	
	vectors, vectors_ix = load_embedding_vectors(args.embedding_path, vocab_dic, reduced_embeddings_path)
	oov = []
	
	#for word in tqdm(vocab, 'extracting oov', total=vocab_size):
	for word in vocab:
		if word not in vectors_ix:
			oov.append([word])
			
	oov_path = os.path.join(os.path.split(args.vocab_path)[0], 'oov_words.txt')
	with open(oov_path, 'w') as f:
		csvwriter = csv.writer(f)
		csvwriter.writerows(oov)
	
	print('done')


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vocab_path',type=lambda x: os.path.expanduser(x))
	parser.add_argument('embedding_path',type=lambda x: os.path.expanduser(x))
	parser.add_argument(
		'--verbosity',
		choices=[
			'DEBUG',
			'ERROR',
			'FATAL',
			'INFO',
			'WARN'
		],
		default='INFO',
	)
	
	run( parser.parse_args())
	