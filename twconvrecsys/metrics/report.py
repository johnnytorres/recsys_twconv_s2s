
import os
import argparse
import pandas as pd
import numpy as np

from twconvrecsys.data.csvreader import DataHandler
from twconvrecsys.metrics.recall import RecallEvaluator


def generate_benchmark(args):
    base_results_dir = args.job_dir
    base_results_dir = os.path.expanduser(base_results_dir)
    results_path = os.path.join(base_results_dir, 'benchmarking.csv')

    data_handler = DataHandler()
    train, valid, test = data_handler.load_data(args.data_dir)

    if os.path.exists(results_path):
        os.remove(results_path)

    print('processing...', end='')

    for d in os.walk(base_results_dir):
        results_dir = d[0]

        if not os.path.isdir(results_dir):
            continue

        path = os.path.join(results_dir, 'predictions.csv')

        if not os.path.exists(path):
            continue

        model = os.path.split(path)[0]
        model = os.path.split(model)[1]
        print(model.upper())

        ds = pd.read_csv(path, header=None)
        col_names = ['target_{}'.format(i) for i in ds.columns]
        ds.columns = col_names

        y_pred = ds.values
        y_true = test.label.values
        metrics = RecallEvaluator.evaluate(y_true, y_pred)

        print(metrics)

        # for model in models:
        #     metricsds = pd.DataFrame.from_dict(metrics[model])
        #     metricsds = metricsds.reset_index()
        #     metricsds.rename(columns={'index': 'metric'}, inplace=True)
        #     metricsds = pd.pivot_table(metricsds, columns='metric')
        #     metricsds = metricsds.reset_index()
        #     metricsds = metricsds.reset_index(drop=True)
        #     metricsds.rename(columns={'index': 'label'}, inplace=True)
        #     metricsds['model'] = model
        #     # print(metricsds)
        #     # metricsds[['model', 'label'] + ['precision','recall','fscore','auc']]
        #     if mds is None:
        #         mds = metricsds
        #     else:
        #         mds = mds.append(metricsds, ignore_index=False)
        #
        # mds['folder'] = os.path.basename(os.path.dirname(path))
        # # fpath = os.path.join(base_results_dir, 'benchmarking.csv')
        # write_header = not os.path.exists(results_path)
        # with open(results_path, 'a') as f:
        #     mds.to_csv(f, index=False, header=write_header)
    print('[OK]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=lambda path: os.path.expanduser(path))
    parser.add_argument('--job-dir', type=lambda path: os.path.expanduser(path))
    generate_benchmark(parser.parse_args())

