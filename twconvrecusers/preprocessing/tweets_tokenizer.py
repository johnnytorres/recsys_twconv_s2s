import os

import argparse
import logging
import unicodedata
import nltk
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
from preprocessor import tokenize, preprocess, clean

nltk.download('stopwords')

import re



class DataTokenizer:
    def __init__(self, args):
        self.args = args
        self.input_file = os.path.expanduser(args.input_file)
        self.output_file = os.path.expanduser(args.output_file)
        self.text_field = args.text_field

        # 0: not tokenize, 1: word_tokenize, 2: punk tokenize, 3: twitter tokenizer, 4: improved twitter tokenizer
        self.tokenizer = args.tokenizer
        self.tweetokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

        if self.tokenizer == 3:
            self.tweet_tokenizer = TweetTokenizer()

        if self.tokenizer == 5:
            self.tweet_tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)

        if self.tokenizer == 6:
            self.tweet_tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)

        self.stem = False
        if self.stem:
            self.stemmer = SnowballStemmer("english") if self.stem else None

        self.lemmatize = False
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer() if self.lemmatize else None

        if self.args.remove_stopwords:
            stopword_list = nltk.corpus.stopwords.words(args.language)
            if 'no' in stopword_list:
                stopword_list.remove('no')
            if 'not' in stopword_list:
                stopword_list.remove('not')

            self.stopword_list = stopword_list

        self.GRUBER_URLINTEXT_PAT = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')


    def tokenize_short_text(self, raw_tweet_text):

        tweet_text = raw_tweet_text
        # tweet_text = tweet_text.strip()
        # tweet_text = unidecode.unidecode(tweet_text)

        if self.args.use_lowercase:
            tweet_text = tweet_text.lower()


        if self.tokenizer > 0:
            if self.tokenizer == 1:
                uttterance_tokens = word_tokenize(tweet_text)
            if self.tokenizer == 2:
                uttterance_tokens = wordpunct_tokenize(tweet_text)
            if self.tokenizer == 3:
                tweet_text = self.GRUBER_URLINTEXT_PAT.sub('URL', tweet_text)
                uttterance_tokens = self.tweet_tokenizer.tokenize(tweet_text)

            if self.tokenizer == 4 or self.tokenizer==10:
                tweet_text = tokenize(tweet_text)
                tweet_text = self.remove_accented_chars(tweet_text)
                uttterance_tokens = self.tweetokenizer.tokenize(tweet_text)
                uttterance_tokens = self.remove_duplicated_sequential_words(uttterance_tokens)
                uttterance_tokens = self.remove_stopwords(uttterance_tokens)

            if self.tokenizer == 5:
                # replace usernames, url, hashtags, numbers, emoticons
                tweet_text = tokenize(' '.join(self.tweet_tokenizer.tokenize(tweet_text)))
                return tweet_text

            if self.tokenizer == 6:
                tweet_text = clean(' '.join(self.tweet_tokenizer.tokenize(tweet_text)))
                return tweet_text

            if self.stem:
                uttterance_tokens = [list(map(self.stemmer.stem, sub)) for sub in uttterance_tokens]
            if self.lemmatize:
                uttterance_tokens = [[self.lemmatizer.lemmatize(tok, pos='v') for tok in sub] for sub in
                                     uttterance_tokens]

            tweet_text = " ".join(uttterance_tokens)

        return tweet_text

    def remove_stopwords(self, tokens):
        if self.args.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopword_list]
        return tokens

    def remove_duplicated_sequential_words(self, uttterance_tokens):
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

    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = text.replace('$', '')
        return text

    def run(self):


        ds = pd.read_csv(
            self.input_file,
            sep=self.args.sep
        )
        desc = os.path.split(self.input_file)[1]
        comments_tokenized = []

        for i, row in tqdm(ds.iterrows(), 'tokenize {}'.format(desc), total=ds.shape[0]):
            comments_tokenized.append(self.tokenize_short_text(row[self.text_field]))


        if self.args.tokenizer==10:
            vectorizer = CountVectorizer(
                strip_accents='ascii',
                stop_words=self.stopword_list,
                max_df=0.9,
                min_df=1,
                max_features=200
            )
            vectorizer.fit(comments_tokenized)
            comments_vec = []

            for tweet in tqdm(comments_tokenized, 'vectorizing', total=len(comments_tokenized)):
                tweet_vec = [word for word in tweet.split() if word in vectorizer.vocabulary_]
                comments_vec.append(' '.join(tweet_vec))

            comments_tokenized = comments_vec



        ds[self.text_field] = comments_tokenized
        output_file = os.path.expanduser(self.output_file)
        ds.to_csv(output_file, index=False, sep=self.args.sep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--output-file', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--sep')
    parser.add_argument('--text-field')
    parser.add_argument('--language')
    parser.add_argument('--tokenizer', type=int, default=4)
    parser.add_argument('--remove-stopwords', action='store_true')
    parser.add_argument('--use-lowercase', action='store_true')
    parser.add_argument('--use-stem', action='store_true')
    parser.add_argument('--use-lemma', action='store_true')

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

    builder = DataTokenizer(parser.parse_args())
    builder.run()
