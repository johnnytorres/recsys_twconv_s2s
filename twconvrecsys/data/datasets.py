
import os
import keras


def prepare_dataset(args):
    if args.data_dir.startswith('gs://') or os.path.exists(args.data_dir):
        expand_dirs(args)
        return

    fname = "{}.zip".format(args.data_dir)
    origin = "https://storage.googleapis.com/ml-research-datasets/twconv/{}".format(fname)
    cache_subdir = "datasets/{}".format(args.data_dir)
    fpath = keras.utils.get_file(fname, origin, cache_subdir=cache_subdir, extract=True)
    args.data_dir = os.path.split(fpath)[0]
    expand_dirs(args)


def expand_dirs(args):

    args.data_dir = os.path.join(args.data_dir, args.data_subdir)

    if args.train_files:
        args.train_files = os.path.join(args.data_dir, args.train_files)
        args.eval_files = os.path.join(args.data_dir, args.eval_files)
        args.test_files = os.path.join(args.data_dir, args.test_files)
        args.vocab_path = os.path.join(args.data_dir, args.vocab_path)

    if args.embedding_path:
        args.embedding_path = os.path.join(args.data_dir, args.embedding_path)

    if not args.job_dir:
        args.job_dir = os.path.join(args.data_dir, 'results', args.estimator)





