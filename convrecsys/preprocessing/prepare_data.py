
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def run(args):
    print('loading data')
    path = args.datafile
    ds = pd.read_csv(path, dtype=object)
    conversations = ds.conv_id.unique()
    conversations = conversations.astype(int)
    conversations = np.sort(conversations)
    print('spliting data')
    X_train, X_test = train_test_split(conversations, test_size=0.25, random_state=args.random_state, shuffle=False)
    X_valid, X_test = train_test_split(X_test, test_size=0.5, random_state=args.random_state, shuffle=False)
    X_train = X_train.astype(str)
    X_valid = X_valid.astype(str)
    X_test = X_train.astype(str)
    X_train = ds[ds.conv_id.isin(X_train)]
    X_valid = ds[ds.conv_id.isin(X_valid)]
    X_test = ds[ds.conv_id.isin(X_test)]
    print('saving data')
    os.makedirs(args.outputdir, exist_ok=True)
    path = os.path.join(args.outputdir, 'train.csv')
    X_train.to_csv(path, index=False)
    path = os.path.join(args.outputdir, 'valid.csv')
    X_valid.to_csv(path, index=False)
    path = os.path.join(args.outputdir, 'test.csv')
    X_test.to_csv(path, index=False)
    print('done!')

if __name__ == '__main__':
    parser  =argparse.ArgumentParser()
    parser.add_argument('datafile', type=lambda x: os.path.expanduser(x))
    parser.add_argument('outputdir', type=lambda x: os.path.expanduser(x))
    parser.add_argument('--random-state',type=int, default=1)
    run(parser.parse_args())






