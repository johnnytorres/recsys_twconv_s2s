import os
import subprocess

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
    data_dir = os.path.join(args.data_dir, args.data_subdir)

    if args.train_files:
        args.train_files = os.path.join(data_dir, args.train_files)
        args.eval_files = os.path.join(data_dir, args.eval_files)
        args.test_files = os.path.join(data_dir, args.test_files)
        args.predict_files = os.path.join(data_dir, args.predict_files) if args.predict_files else None
        args.vocab_path = os.path.join(data_dir, args.vocab_path)
        args.vocab_processor_path = os.path.join(data_dir, 'vocab_processor.bin')

    if args.embedding_path and args.embedding_enabled:
        args.embedding_path = os.path.join(args.data_dir, args.embedding_path)
    else:
        args.embedding_path = None

    if not args.job_dir:
        args.job_dir = os.path.join(data_dir, 'results', args.estimator)

    # get train size
    train_csvrecords = os.path.join(data_dir, 'train.csvrecords')
    args.train_size = wccount(train_csvrecords)


def wccount(filename):
    print('counting lines in file {}'.format(filename))
    out = subprocess.Popen(
        ['wc', '-l', filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    ).communicate()[0]
    print('TRAIN SIZE', out)
    num_instances=int(out.split()[0])-1 # minus header
    return num_instances
