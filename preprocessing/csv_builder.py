import argparse
import csv
import os
import random
import tarfile
import unicodedata

import nltk
import unicodecsv
import unidecode
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from preprocessor import tokenize, preprocess
from six.moves import urllib
from tqdm import tqdm

dialog_end_symbol = "__dialog_end__"
end_of_utterance_symbol = "__eou__"
end_of_turn_symbol = "__eot__"


stemmer = SnowballStemmer("spanish")
lemmatizer = WordNetLemmatizer()
tweetokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

stopword_list = nltk.corpus.stopwords.words('spanish')
#stopword_list.remove('no')
#stopword_list.remove('not')
#from preprocessor.preprocess import Preprocess


profiles = {}
profiles_names = []

def load_profiles(args):
	global profiles, profiles_names
	fpath = args.profiles_path
	
	with open(fpath, 'r') as f:
		reader  = csv.reader(f)
		next(reader)
		for row in tqdm( reader, 'loading profiles'):
			profiles[row[0]] =  row[1]
			
	profiles_names = list(profiles.keys())

def translate_dialog_to_lists(dialog_filename):
	"""
	Translates the dialog to a list of lists of utterances. In the first
	list each item holds subsequent utterances from the same user. The second level
	list holds the individual utterances.
	:param dialog_filename:
	:return:
	"""
	
	with open(dialog_filename, 'r') as dialog_file:
		# dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t',quoting=csv.QUOTE_NONE)
		dialog_reader = csv.reader(dialog_file, delimiter='\t', quoting=csv.QUOTE_NONE)
		
		# go through the dialog
		first_turn = True
		dialog = []
		users = []
		same_user_utterances = []
		# last_user = None
		dialog.append(same_user_utterances)
		
		for dialog_line in dialog_reader:
			
			if first_turn:
				last_user = dialog_line[1]
				first_turn = False
			
			if last_user != dialog_line[1]:
				# user has changed
				same_user_utterances = []
				dialog.append(same_user_utterances)
			
			tweet_text =dialog_line[args.text_field]
			
			utterance_text = tokenize_utterance(tweet_text, args)
			
			same_user_utterances.append(utterance_text)
			
			last_user = dialog_line[1]
			users.append(last_user)
		
		dialog.append([dialog_end_symbol])
	
	return list(zip(users,dialog))


def tokenize_utterance(tweet_text, args):
	utterance_text = unidecode.unidecode(tweet_text)
	if args.lowercase:
		utterance_text = utterance_text.lower()
	if args.tokenize or args.tokenize_punk or args.tokenize_tweets:
		
		if args.tokenize_tweets:
			utterance_text = tokenize(utterance_text)
			# todo: expand contractions
			utterance_text = remove_accented_chars(utterance_text)
			uttterance_tokens = tweetokenizer.tokenize(utterance_text)
			uttterance_tokens = remove_duplicated_sequential_words(uttterance_tokens)
		
		elif args.tokenize:
			uttterance_tokens = nltk.word_tokenize(utterance_text)
		elif args.tokenize_punk:
			uttterance_tokens = nltk.wordpunct_tokenize(utterance_text)
		
		if args.remove_stopwords:
			uttterance_tokens = remove_stopwords(uttterance_tokens)
		
		if args.stem:
			uttterance_tokens = [list(map(stemmer.stem, sub)) for sub in uttterance_tokens]
		if args.lemmatize:
			uttterance_tokens = [[lemmatizer.lemmatize(tok, pos='v') for tok in sub] for sub in uttterance_tokens]
		
		utterance_text = " ".join(uttterance_tokens)
	return utterance_text


def remove_duplicated_sequential_words(uttterance_tokens):
	i = 0
	while i < len(uttterance_tokens):
		j = i + 1
		while j < len(uttterance_tokens):
			if uttterance_tokens[i] == uttterance_tokens[j]:
				del uttterance_tokens[j]
			else:
				break
		i += 1
	return uttterance_tokens
		
def remove_accented_chars(text):
	text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	text = text.replace('$', '')
	return text
	
def remove_stopwords(tokens):
	filtered_tokens = [token for token in tokens if token not in stopword_list]
	return filtered_tokens

def get_random_utterances_from_corpus(candidate_dialog_paths, rng, utterances_num=9, min_turn=3, max_turn=20):
	"""
	Sample multiple random utterances from the whole corpus.
	:param candidate_dialog_paths:
	:param rng:
	:param utterances_num: number of utterances to generate
	:param min_turn: minimal index of turn that the utterance is selected from
	:return:
	"""
	utterances = []
	dialogs_num = len(candidate_dialog_paths)
	
	for i in range(0, utterances_num):
		# sample random dialog
		dialog_path = candidate_dialog_paths[rng.randint(0, dialogs_num - 1)]
		# load the dialog
		dialog = translate_dialog_to_lists(dialog_path)
		
		# we do not count the last  _dialog_end__ urn
		dialog_len = len(dialog) - 1
		#if (dialog_len < min_turn):
			#print("Dialog {} was shorter than the minimum required lenght {}".format(dialog_path, dialog_len))
			# exit()
		# sample utterance, exclude the last round that is always "dialog end"
		max_ix = min(max_turn, dialog_len) - 1
		
		# this code deals with corner cases when dialogs are too short
		if max_ix <= min_turn - 1:
			turn_index = max_ix
		else:
			turn_index = rng.randint(min_turn, max_ix)
		
		utterance = singe_user_utterances_to_string(dialog[turn_index])
		utterances.append(utterance)
	return utterances


def singe_user_utterances_to_string(utterances_list):
	"""
	Concatenates multiple user's utterances into a single string.
	:param utterances_list:
	:return:
	"""
	return " ".join([x + " " + end_of_utterance_symbol for x in utterances_list])


def dialog_turns_to_string(dialog):
	"""
	Translates the whole dialog (list of lists) to a string
	:param dialog:
	:return:
	"""
	# join utterances
	user, dialog=zip(*dialog)
	turns_as_strings = list(map(singe_user_utterances_to_string, dialog))
	# join turns
	return "".join([x + " " + end_of_turn_symbol + " " for x in turns_as_strings])


def create_random_context(dialog, rng, min_context_length=2, max_context_length=20):
	"""
	Samples random context from a dialog. Contexts are uniformly sampled from the whole dialog.
	:param dialog:
	:param rng:
	:return: context, index of next utterance that follows the context
	"""
	# sample dialog context
	# context_turns = rng.randint(minimum_context_length,len(dialog)-1)
	max_len = min(max_context_length, len(dialog)) - 2
	if max_len <= min_context_length:
		context_turns = max_len
	else:
		context_turns = rng.randint(min_context_length, max_len)
	
	# create string
	return dialog_turns_to_string(dialog[:context_turns]), context_turns

def get_positive_user_profile(dialog, next_utterance_ix):
	screen_name = dialog[next_utterance_ix][0]
	
	#todo: filter tweets that already are in the conversation
	if screen_name in profiles:
		return profiles[screen_name]
	
	return None
	
def get_negative_user_profiles(num_profiles):
	uprofiles = random.sample(profiles_names, num_profiles)
	response = []
	
	#todo: filter users that already are in the conversation
	for screen_name in uprofiles:
		response.append( profiles[screen_name])
		
	return response
	
	
def create_single_dialog_train_example(dialog, candidate_dialog_paths, rng, positive_probability,
                                       min_context_length=2, max_context_length=20):
	"""
	Creates a single example for training set.
	:param context_dialog_path:
	:param candidate_dialog_paths:
	:param rng:
	:param positive_probability:
	:return:
	"""
	
	context_str, next_utterance_ix = create_random_context(dialog, rng,
	                                                       min_context_length=min_context_length,
	                                                       max_context_length=max_context_length)
	
	#global profiles
	prob = rng.random()
	if positive_probability > prob:
		# use the next utterance as positive example
		response = get_positive_user_profile(dialog, next_utterance_ix)
		label = 1
	else:
		response = get_negative_user_profiles(num_profiles=1)[0]
		label = 0
		
	return context_str, response, label





def create_single_dialog_test_example(dialog, candidate_dialog_paths, rng, distractors_num,
                                      min_context_length=2, max_context_length=20):
	"""
	Creates a single example for testing or validation. Each line contains a context, one positive example and N negative debug.
	:param context_dialog_path:
	:param candidate_dialog_paths:
	:param rng:
	:param distractors_num:
	:return: triple (context, positive response, [negative responses])
	"""
	
	context_str, next_utterance_ix = create_random_context(
		dialog, rng, min_context_length=min_context_length,
		max_context_length=max_context_length)
	
	positive_response = get_positive_user_profile(dialog, next_utterance_ix)
	negative_responses = get_negative_user_profiles(distractors_num)
	return context_str, positive_response, negative_responses


def create_examples(candidate_dialog_paths, examples_num, creator_function):
	"""
	Creates a list of training debug from a list of dialogs and function that transforms a dialog to an example.
	:param candidate_dialog_paths:
	:param creator_function:
	:return:
	"""
	examples = []
	missing= 0
	unique_dialogs_num = len(candidate_dialog_paths)
	
	for i in tqdm(range(examples_num), 'creating examples', total=examples_num):
		context_dialog_path = candidate_dialog_paths[i % unique_dialogs_num]
		context_dialog = translate_dialog_to_lists(context_dialog_path)
		# without diag end utt, validate min context length
		if len(context_dialog) - 1 < args.min_context_length:
			continue
		
		example = creator_function(context_dialog, candidate_dialog_paths)
		
		if example[1] is None: # ground truth user profile not found
			missing += 1
			continue
		
		examples.append(example)
	
	return examples


def convert_csv_with_dialog_paths(csv_file):
	"""
	Converts CSV file with comma separated paths to filesystem paths.
	:param csv_file:
	:return:
	"""
	
	def convert_line_to_path(line):
		path_info = [x.strip() for x in line.split(",")]
		
		if len(path_info) > 1:
			file, dir = path_info
			return file if not dir else dir +'/'+ file
		else:
			return path_info[0]
	
	return list(map(convert_line_to_path, csv_file))


def prepare_data_maybe_download(directory):
	"""
  Download and unpack dialogs if necessary.
  """
	filename = 'ubuntu_dialogs.tgz'
	url = 'http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz'
	dialogs_path = os.path.join(directory, 'dialogs')
	
	# test it there are some dialogs in the path
	if not os.path.exists(os.path.join(directory, "10", "1.tst")):
		# dialogs are missing
		archive_path = os.path.join(directory, filename)
		if not os.path.exists(archive_path):
			# archive missing, download it
			print(("Downloading %s to %s" % (url, archive_path)))
			filepath, _ = urllib.request.urlretrieve(url, archive_path)
			print("Successfully downloaded " + filepath)
		
		# unpack data
		if not os.path.exists(dialogs_path):
			print("Unpacking dialogs ...")
			with tarfile.open(archive_path) as tar:
				tar.extractall(path=directory)
			print("Archive unpacked.")
		
		return


#####################################################################################
# Command line script related code
#####################################################################################

if __name__ == '__main__':
	
	def create_eval_dataset(args, file_list_csv):
		
		print(f'generating {args.output}')
		load_profiles(args)
		rnd = random.Random(args.seed)
		dialog_paths = get_dialog_paths(args, file_list_csv)
		args.examples = get_num_examples(args, dialog_paths)
		
		data_set = create_examples(
			dialog_paths,
			args.examples,
			lambda context_dialog, candidates:
			create_single_dialog_test_example(
				context_dialog, candidates, rnd, args.n,
				min_context_length=args.min_context_length,
				max_context_length=args.max_context_length))
		
		# output the dataset
		with open(args.output, 'wb') as f:
			w = unicodecsv.writer(f, encoding='utf-8')
			header = ["Context", "Response"]
			header.extend(["Distractor_{}".format(x) for x in range(args.n)])
			w.writerow(header)
			#print(f'writing {len(data_set)} examples...')
			for row in tqdm(data_set, f'writing'):
				translated_row = [row[0], row[1]]
				translated_row.extend(row[2])
				w.writerow(translated_row)
			#print(("Dataset stored in: {}".format(args.output)))
	
	
	def get_dialog_paths(args, file_list_csv):
		f = open(os.path.join(args.data_root, file_list_csv), 'r')
		dialog_paths = [args.data_root +'/'+ path for path in tqdm(convert_csv_with_dialog_paths(f),'loading dialog paths')]
		return dialog_paths
	
	
	def train_cmd(args):
		print(f'generating {args.output}')
		
		load_profiles(args)
		rnd = random.Random(args.seed)
		dialog_paths = get_dialog_paths(args, "trainfiles.csv")
		args.examples = get_num_examples(args, dialog_paths)
		
		train_set = create_examples(
			dialog_paths,
			args.examples,
			lambda context_dialog, candidates:
			create_single_dialog_train_example(
				context_dialog, candidates, rnd, args.p,
				min_context_length=args.min_context_length,
				max_context_length=args.max_context_length))
		
		# output the dataset
		with open(args.output, 'wb') as f:
			w = unicodecsv.writer(f, encoding='utf-8')
			w.writerow(["Context", "Utterance", "Label"])
			#print(f'writing {len(train_set)} examples...')
			w.writerows(train_set)
			#print(("Train dataset stored in: {}".format(args.output)))
	
	
	def get_num_examples(args, dialog_paths):
		return len(dialog_paths) if args.examples == -1 else args.examples
	
	
	def valid_cmd(args):
		create_eval_dataset(args, "valfiles.csv")
	
	
	def test_cmd(args):
		create_eval_dataset(args, "testfiles.csv")
	
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	                                 description="Script that creates train, valid and test set from 1 on 1 dialogs in Ubuntu Corpus. " +
	                                             "The script downloads 1on1 dialogs from internet and then it randomly samples all the datasets with positive and negative debug.")
	
	parser.add_argument(
		'--data-root',
		type=lambda x: os.path.expanduser(x),
		default='data/convusersec/dialogs',
	    help='directory where 1on1 dialogs will be downloaded and extracted, the data will be downloaded from cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz')

	parser.add_argument(
		'--profiles-path',
		type=lambda x: os.path.expanduser(x),
        default='data/convusersec/profiles.csv',
        help='file with the profiles'
	)
	
	parser.add_argument(
		'-o',
		'--output',
		type=lambda x: os.path.expanduser(x),
		default=None,
		help='output csv'
	)


	parser.add_argument('--seed', type=int, default=1234,
	                    help='seed for random number generator')
	
	parser.add_argument('--text-field', type=int, default=3,
	                    help='text field index in dialogs (default ubuntu (3), twconv (2)')
	
	parser.add_argument('--min-context-length', type=int, default=3,
	                    help='maximum number of dialog turns in the context')
	
	parser.add_argument('--max-context-length', type=int, default=10,
	                    help='maximum number of dialog turns in the context')
	
	
	#----------------------------------------------------
	# preprocessing args
	parser.add_argument('-t', '--tokenize', action='store_true', help='tokenize the output')
	parser.add_argument('-tp', '--tokenize-punk', action='store_true', help='tokenize the output using wordpunct_tokenize')
	parser.add_argument('-tt', '--tokenize-tweets', action='store_true', help='tokenize the output using tweets tokenizer')
	
	parser.add_argument('-lc', '--lowercase', action='store_true', default=True,
	                    help='lowercase the output')
	parser.add_argument('-rs', '--remove-stopwords', action='store_true', default=False,
	                    help='lowercase the output')
	parser.add_argument('-s', '--stem', action='store_true',
	                    help='stem the output by nltk.stem.SnowballStemmer (applied only when -t flag is present)')
	parser.add_argument('-l', '--lemmatize', action='store_true',
	                    help='lemmatize the output by nltk.stem.WorldNetLemmatizer (applied only when -t flag is present)')
	# ----------------------------------------------------
	
	parser.add_argument('-e', '--examples', type=int, default=-1, help='number of examples to generate')
	subparsers = parser.add_subparsers(help='sub-command help')
	
	parser_train = subparsers.add_parser('train', help='trainset generator')
	parser_train.add_argument('-p', type=float, default=0.5, help='positive example probability')
	
	parser_train.set_defaults(func=train_cmd)
	
	parser_test = subparsers.add_parser('test', help='testset generator')
	parser_test.add_argument('-n', type=int, default=9, help='number of distractor debug for each context')
	parser_test.set_defaults(func=test_cmd)
	
	parser_valid = subparsers.add_parser('valid', help='validset generator')
	parser_valid.add_argument('-n', type=int, default=9, help='number of distractor debug for each context')
	parser_valid.set_defaults(func=valid_cmd)
	
	args = parser.parse_args()
	args.func(args)
