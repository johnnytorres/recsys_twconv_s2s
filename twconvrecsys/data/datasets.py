
import os
import keras


class DatasetNames:
    TREC='2011_trec'
    TREC_SAMPLE='2011_trec_sample'


def prepare_dataset(args):
    if args.data_dir.startswith('gs://') or os.path.exists(args.data_dir):
        expand_dirs(args)
        return

    if args.data_dir == DatasetNames.TREC or args.data_dir == DatasetNames.TREC_SAMPLE:
        fname = "twconv_2011_trec.v1.zip"
        origin = "https://storage.googleapis.com/ml-research-datasets/twconv/{}".format(fname)
        cache_subdir = "datasets/twconv_2011_trec"
        fpath = keras.utils.get_file(fname, origin, cache_subdir=cache_subdir, extract=True)
        if args.data_dir == DatasetNames.TREC_SAMPLE:
            args.data_dir = os.path.join(os.path.split(fpath)[0], 'sampledata')
        else:
            args.data_dir = os.path.join(os.path.split(fpath)[0], 'staggingdata')

    expand_dirs(args)


def expand_dirs(args):
    args.train_files = os.path.join(args.data_dir, args.train_files)
    args.eval_files = os.path.join(args.data_dir, args.eval_files)
    args.test_files = os.path.join(args.data_dir, args.test_files)
    args.vocab_path = os.path.join(args.data_dir, args.vocab_path)
    if args.embedding_path:
        args.embedding_path = os.path.join(args.data_dir, args.embedding_path)




