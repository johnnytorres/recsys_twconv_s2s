
import argparse
import csv
import os
import pandas as pd
from tqdm import tqdm
from twconvrecusers.preprocessing.dialogs_builder import tokenize_utterance


def build_profiles_from_timeline(args):

	input_path = args.input
	output_path = args.output

	profiles = {}

	with open(input_path, 'r') as f:

		reader = csv.DictReader(f)

		prior_screen_name = ''

		tweets = []

		for ix, row in tqdm( enumerate(reader)):

			screen_name = row['screen_name']

			if prior_screen_name != screen_name:
				tl_tweets = tweets[-args.n:]
				tl_tweets = [t.replace('\r', ' ').replace('\n', ' ') for t in tl_tweets]
				tweet_text = ' '.join(tl_tweets)
				profiles[prior_screen_name] = tweet_text
				prior_screen_name = screen_name
				tweets =[]

				# if ix >= 10:
				# 	break

			tweet_text = row['text']
			tweets.append(tweet_text)


	del profiles['']

	ds = pd.DataFrame.from_dict(profiles, orient='index')
	ds = ds.reset_index()
	ds.columns  = ['screen_name' , 'timeline']
	ds.to_csv(output_path, index=False)


def build_profiles_from_conversations(args):
	input_path = args.input # this is the dialogs datadir
	split_names = ['train', 'val', 'test']
	for split in split_names:
		print(f'building {split} profiles...', end='')
		path = os.path.join(input_path, f'{split}files.csv')
		diag_ids = []
		with open(path, 'r') as f:
			for diag_id in f:
				diag_ids.append(diag_id.split('.')[0])
		path = os.path.join(input_path, 'dialogs.csv')
		conv = pd.read_csv(path, dtype=object, usecols=['dialog_id', 'screen_name', 'text'])
		conv = conv[conv.dialog_id.isin(diag_ids)]
		users = conv.groupby('screen_name')['text'].apply(lambda x: ' '.join(x))
		users = users.reset_index()
		users.columns = ['screen_name', 'timeline']
		path = os.path.join(input_path, f'profiles_{split}.csv')
		users.to_csv(path, index=False)
		print('done!')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=lambda x: os.path.expanduser(x), help='timelines file or dialogs dir')
	parser.add_argument('--output', type=lambda x: os.path.expanduser(x))
	parser.add_argument('--n', type=int, default=10, help='number of tweets from the timeline')
	
	# ----------------------------------------------------
	# preprocessing args
	parser.add_argument('-t', '--tokenize', action='store_true', help='tokenize the output')
	parser.add_argument('-tp', '--tokenize-punk', action='store_true',
						help='tokenize the output using wordpunct_tokenize')
	parser.add_argument('-tt', '--tokenize-tweets', action='store_true',
	                    help='tokenize the output using tweets tokenizer')
	parser.add_argument('-lc', '--lowercase', action='store_true', #default=True,
	                    help='lowercase the output')
	parser.add_argument('-rs', '--remove-stopwords', action='store_true', #default=False,
	                    help='lowercase the output')
	parser.add_argument('-s', '--stem', action='store_true',
	                    help='stem the output by nltk.stem.SnowballStemmer (applied only when -t flag is present)')
	parser.add_argument('-l', '--lemmatize', action='store_true',
	                    help='lemmatize the output by nltk.stem.WorldNetLemmatizer (applied only when -t flag is present)')
	# ----------------------------------------------------
	subparsers = parser.add_subparsers(help='sub-command help')
	parser_bp = subparsers.add_parser('buildfromtl', help='build profile from timeline')
	parser_bp.set_defaults(func=build_profiles_from_timeline)

	parser_bp = subparsers.add_parser('buildfromconv', help='build profile from conversations')

	parser_bp.set_defaults(func=build_profiles_from_conversations)
	args = parser.parse_args()

	# ISSUE with only urls
	
	# from html.parser import HTMLParser
	#
	# from bs4 import BeautifulSoup as Soup
	# import re
	#
	# GRUBER_URLINTEXT_PAT = re.compile(
	# 	r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
	#
	# with open('data/issue.txt', 'r') as f:
	# 	tweet_text = f.read()
	# 	#urls = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', tweet_text)
	# 	urls = GRUBER_URLINTEXT_PAT.findall(tweet_text)
	# 	#html = Soup(tweet_text, 'html.parser')
	# 	#for u in urls:
	# 		# tweet_text = tweet_text.rep
	# 	tweet_text = GRUBER_URLINTEXT_PAT.sub( '', tweet_text)
	#
	# 	tokenize_utterance(tweet_text,args )
	
	args.func(args)