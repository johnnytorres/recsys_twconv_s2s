
import os
import argparse
import pandas as pd

from twconvrecsys.data.csvreader import DataHandler
from twconvrecsys.metrics.ndgc import NdgcEvaluator
from twconvrecsys.metrics.recall import RecallEvaluator
from twconvrecsys.metrics.precision import PrecisionEvaluator


def generate_benchmark(args):
    base_results_dir = args.job_dir
    base_results_dir = os.path.expanduser(base_results_dir)
    results_path = os.path.join(base_results_dir, 'benchmarking.csv')

    data_handler = DataHandler()
    test = data_handler.load_test_data(args)

    if os.path.exists(results_path):
        os.remove(results_path)

    print('processing...')

    benchmark_ds = None

    for d in os.walk(base_results_dir):
        results_dir = d[0]

        if not os.path.isdir(results_dir):
            continue

        path = os.path.join(results_dir, 'predictions.csv')

        if not os.path.exists(path):
            continue

        dataset, model = os.path.split(results_dir)
        _, dataset = os.path.split(dataset)
        print(f'{dataset.upper()} - {model.upper()}')

        ds = pd.read_csv(path, header=None)
        col_names = ['target_{}'.format(i) for i in ds.columns]
        ds.columns = col_names

        y_pred = ds.values
        y_true = test.label.values
        metrics = RecallEvaluator.calculate(y_true, y_pred)
        metrics_ds = pd.DataFrame(metrics, columns=['metric', 'k', 'N', 'value'])

        metrics = PrecisionEvaluator.calculate(y_true, y_pred)
        metrics_ds = metrics_ds.append(pd.DataFrame(metrics, columns=['metric', 'k', 'N', 'value']), ignore_index=True)

        metrics = NdgcEvaluator.calculate(y_true, y_pred)
        metrics_ds = metrics_ds.append(pd.DataFrame(metrics, columns=['metric', 'k', 'N', 'value']), ignore_index=True)

        metrics_ds['dataset'] = dataset
        metrics_ds['model'] = model

        benchmark_ds = metrics_ds if benchmark_ds is None else benchmark_ds.append(metrics_ds, ignore_index=True)

    cols = ['dataset', 'model', 'metric', 'k', 'N', 'value']
    benchmark_ds[cols].to_csv(results_path, index=False)
    print('[OK]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=lambda path: os.path.expanduser(path))
    parser.add_argument('--data-subdir', type=lambda path: os.path.expanduser(path))
    parser.add_argument('--job-dir', type=lambda path: os.path.expanduser(path))
    generate_benchmark(parser.parse_args())

