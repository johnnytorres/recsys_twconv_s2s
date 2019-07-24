import argparse
import csv
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


class DialogsBuilder:
    def __init__(self, args):
        self.args = args
        self.num_distractors = 9

    def run(self):
        path = os.path.join(self.args.data_dir, 'dialogs.csv')
        ds = pd.read_csv(path)
        dialogs = ds.groupby('dialog_id')

        trainds = None
        validds = None
        testds = None

        # split dataset
        count = 0
        for diag_id, dialog in tqdm(dialogs, total=len(dialogs)):
            test_size = np.maximum(int(np.round(dialog.shape[0] * self.args.test_size)), 1)
            train_ix = dialog.shape[0] - (test_size * 2)  # valid and test sets
            valid_ix = train_ix + test_size

            train_dialog = dialog.iloc[:train_ix, :]
            valid_dialog = dialog.iloc[train_ix:valid_ix, :]
            test_dialog = dialog.iloc[valid_ix:, :]

            trainds = train_dialog if trainds is None else trainds.append(train_dialog, ignore_index=True)
            validds = valid_dialog if validds is None else validds.append(valid_dialog, ignore_index=True)
            testds = test_dialog if testds is None else testds.append(test_dialog, ignore_index=True)

            # count+=1
            # if count > 10:
            #     break

        # ----------------------
        # create train set in format (conv, user profile, flag)
        # ----------------------
        userprofiles = trainds.groupby('screen_name')['text'].apply(lambda t: ' '.join(t))
        # TODO: first approach use all tweets, several context can be generated for long dialogs
        diagcontexts = trainds.groupby('dialog_id')#['text'].apply(lambda t: ' '.join(t))

        train_instances = []
        valid_instances = []

        for id, dialog in diagcontexts:
            dialog_context = dialog['text'].str.cat(sep=' ')
            negative_profiles =userprofiles[~userprofiles.index.isin(dialog.screen_name.unique()) ]
            for ix, username in dialog.screen_name.iteritems():
                user_profile = userprofiles[username]
                train_instances.append([dialog_context, user_profile, 1])
                negative_profile = negative_profiles.sample(1, random_state=self.args.seed).values[0]
                train_instances.append([dialog_context, negative_profile, 0])
        # ----------------------

            # create valid and test set in format (conv, user profile, distractor1.. distractorN)
            validtweets = validds[validds.dialog_id==id]
            users_filter = set(list(dialog.screen_name.unique()) + list(validtweets.screen_name.unique()))
            negative_profiles = userprofiles[~userprofiles.index.isin(users_filter)]
            for ix, row in validtweets.iterrows():
                username = row['screen_name']
                user_profile = userprofiles[username] if username in userprofiles else f'NoProfileFor{username}'
                instance = [dialog_context, user_profile]
                distractors = negative_profiles.sample(self.num_distractors, random_state=self.args.seed).values
                instance.extend(distractors)
                valid_instances.append(instance)
                dialog_context += ' ' + row['text']
        
        # save dialogs split
        print('saving dataset split...', end='')
        path = os.path.join(self.args.data_dir, 'train_dialogs.csv')
        trainds.to_csv(path, index=False)
        path = os.path.join(self.args.data_dir, 'valid_dialogs.csv')
        validds.to_csv(path, index=False)
        path = os.path.join(self.args.data_dir, 'test_dialogs.csv')
        testds.to_csv(path, index=False)

        # save dialog in training format
        cols = ['context','profile','flag']
        trainds = pd.DataFrame(train_instances, columns=cols)
        path = os.path.join(self.args.data_dir, 'train.csv')
        trainds.to_csv(path, index=False)
        # save dialogs in validation format
        cols = ['context','profile'] + [f'distractor{i}' for i in range(self.num_distractors)]
        validds = pd.DataFrame(valid_instances, columns=cols)
        path = os.path.join(self.args.data_dir, 'valid.csv')
        validds.to_csv(path, index=False)
        print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-dir',
        type=lambda x: os.path.expanduser(x))

    parser.add_argument(
        '--seed', type=int, default=None,
        help='seed for random number generator')

    parser.add_argument(
        '--text-field', type=str, default='text',
        help='text field index in dialogs ')

    parser.add_argument(
        '--test-size', type=float, default=0.1,
        help='test set size for splitting conversations')

    # parser.add_argument(
    #     '--min-context-length', type=int, default=3,
    #     help='maximum number of dialog turns in the context')

    # parser.add_argument(
    #     '--max-context-length', type=int, default=10,
    #     help='maximum number of dialog turns in the context')

    builder = DialogsBuilder(args = parser.parse_args())
    builder.run()
