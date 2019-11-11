
import os
import keras


class DatasetNames:
    TREC='2011_trec'


def prepare_dataset(args):
    if args.dataset_name == DatasetNames.TREC:
        if not args.data_dir:
            fname = "2011_trec.v6.zip"
            origin = "https://storage.googleapis.com/ml-research-datasets/twconv/{}".format(fname)
            cache_subdir = "datasets/twconv_2011_trec"
            fpath = keras.utils.get_file(fname, origin, cache_subdir=cache_subdir, extract=True)
            args.data_dir = os.path.join(os.path.split(fpath)[0], 'staggingdata')
            # twconv/2011_trec/sampleresults/tfidf
            #args.job_dir = os.path.join(args.job_dir, 'recsys', 'twconv/2011_trec/sampleresults', args.estimator)



