import os
import csv
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from twconvrecsys.data import input, metadata
from twconvrecsys.data.csvreader import DataHandler
from twconvrecsys.data.datasets import prepare_dataset, DatasetNames
from twconvrecsys.metrics.recall import RecallEvaluator
from twconvrecsys.models import neural
from twconvrecsys.models import mf
from twconvrecsys.models import nmf
from twconvrecsys.models.factory import get_model, ModelName
from twconvrecsys.parameters import initialise_params


def clean_job_dir():
    # If job_dir_reuse is False then remove the job_dir if it exists
    tf.compat.v1.logging.info(("Resume training:", HYPER_PARAMS.reuse_job_dir))
    if not HYPER_PARAMS.reuse_job_dir:
        if tf.io.gfile.exists(HYPER_PARAMS.job_dir):
            tf.io.gfile.rmtree(HYPER_PARAMS.job_dir)
            tf.compat.v1.logging.info(("Deleted job_dir {} to avoid re-use".format(HYPER_PARAMS.job_dir)))
        else:
            tf.compat.v1.logging.info("No job_dir available to delete")
    else:
        tf.compat.v1.logging.info(("Reusing job_dir {} if it exists".format(HYPER_PARAMS.job_dir)))


def get_train_input_fn():
    train_input_fn = input.generate_input_fn(
        file_names_pattern=HYPER_PARAMS.train_files,
        HYPER_PARAMS=HYPER_PARAMS,
        file_encoding=HYPER_PARAMS.file_encoding,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_epochs=HYPER_PARAMS.num_epochs,
        batch_size=HYPER_PARAMS.train_batch_size
    )
    return train_input_fn


def get_eval_input_fn():
    eval_input_fn = input.generate_input_fn(
        file_names_pattern=HYPER_PARAMS.eval_files,
        HYPER_PARAMS=HYPER_PARAMS,
        file_encoding=HYPER_PARAMS.file_encoding,
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=HYPER_PARAMS.eval_batch_size
    )
    return eval_input_fn


def get_test_input_fn():
    test_input_fn = input.generate_input_fn(
        file_names_pattern=HYPER_PARAMS.test_files,
        HYPER_PARAMS=HYPER_PARAMS,
        file_encoding=HYPER_PARAMS.file_encoding,
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=HYPER_PARAMS.eval_batch_size
    )
    return test_input_fn


def get_predict_input_fn():
    predict_input_fn = input.generate_input_fn(
        file_names_pattern=HYPER_PARAMS.predict_files,
        file_encoding=HYPER_PARAMS.file_encoding,
        mode=tf.estimator.ModeKeys.TRAIN,
        batch_size=HYPER_PARAMS.predict_batch_size
    )
    return predict_input_fn


def train_model(run_config):
    """Train, evaluate, and export the model using tf.estimator.train_and_evaluate API"""

    train_input_fn = get_train_input_fn()

    eval_input_fn = get_eval_input_fn()

    exporter = tf.estimator.FinalExporter(
        'estimator',
        #input.SERVING_FUNCTIONS[HYPER_PARAMS.export_format],
        input.get_serving_function(HYPER_PARAMS),
        as_text=False  # change to true if you want to export the model as readable text
    )

    hooks = []
    if HYPER_PARAMS.debug:
        hooks = [tf_debug.LocalCLIDebugHook()]

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=HYPER_PARAMS.train_steps,
        hooks=hooks
    )

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=HYPER_PARAMS.eval_steps,
        #exporters=[exporter],
        name='training',
        throttle_secs=HYPER_PARAMS.eval_every_secs,
        hooks=hooks
    )

    tf.compat.v1.logging.info("===========================")
    tf.compat.v1.logging.info("* TRAINING configurations")
    tf.compat.v1.logging.info("===========================")
    tf.compat.v1.logging.info(("Train size: {}".format(HYPER_PARAMS.train_size)))
    tf.compat.v1.logging.info(("Epoch count: {}".format(HYPER_PARAMS.num_epochs)))
    tf.compat.v1.logging.info(("Train batch size: {}".format(HYPER_PARAMS.train_batch_size)))
    tf.compat.v1.logging.info(("Training steps: {} ({})".format(int(HYPER_PARAMS.train_steps),
                                                      "supplied" if HYPER_PARAMS.train_size is None else "computed")))
    tf.compat.v1.logging.info(("Evaluate every {} seconds".format(HYPER_PARAMS.eval_every_secs)))
    tf.compat.v1.logging.info("===========================")

    estimator = get_estimator(run_config)

    # train and evaluate
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )


def get_estimator(run_config):
    if HYPER_PARAMS.estimator == "mf":
        estimator = mf.create_estimator(
            config=run_config,
            HYPER_PARAMS=HYPER_PARAMS
        )
    elif HYPER_PARAMS.estimator == "nmf":
        estimator = nmf.create_estimator(
            config=run_config,
            HYPER_PARAMS=HYPER_PARAMS
        )
    else:
        estimator = neural.create_estimator(
            config=run_config,
            HYPER_PARAMS=HYPER_PARAMS
        )
    return estimator


def test_model(run_config):
    # EVALUATE MODEL WITH TEST DATA
    test_input_fn = get_test_input_fn()
    tf.compat.v1.logging.info("===========================")
    tf.compat.v1.logging.info("* TESTING configurations")
    tf.compat.v1.logging.info("===========================")
    tf.compat.v1.logging.info(("Test batch size: {}".format(HYPER_PARAMS.eval_batch_size)))
    tf.compat.v1.logging.info(("Test steps: {} ({})".format(None, "computed (all tests instances)")))
    tf.compat.v1.logging.info("===========================")

    estimator = get_estimator(run_config)

    hooks = []
    if HYPER_PARAMS.debug:
        hooks = [tf_debug.LocalCLIDebugHook()]

    estimator.evaluate(
        input_fn=test_input_fn,
        hooks=hooks,
        name='tests'
    )

    predictions = estimator.predict(input_fn=test_input_fn, yield_single_examples=False)
    num_targets = HYPER_PARAMS.num_distractors + 1
    path = os.path.join(HYPER_PARAMS.job_dir, 'predictions.csv')

    with tf.io.gfile.GFile(path, 'w') as f:
        csvwriter = csv.writer(f)
        for i, instance_prediction in enumerate(predictions, start=1):
            probs = instance_prediction['logits']
            probs = np.split(probs, num_targets, 0)
            probs = np.array(probs)
            probs = np.concatenate(probs, axis=1)
            probs = probs.tolist()
            # if i % num_instances == 0:
            print(len(probs))
            csvwriter.writerows(probs)
            # if i % 1000 == 0:
            #     print('predicting {} instances'.format(i), end='\r')


def predict_instances(run_config):
    # PREDICT EXAMPLE INSTANCES
    tf.compat.v1.logging.info("===========================")
    tf.compat.v1.logging.info("* PREDICT configurations")
    tf.compat.v1.logging.info("===========================")
    tf.compat.v1.logging.info(("Predict batch size: {}".format(HYPER_PARAMS.predict_batch_size)))
    tf.compat.v1.logging.info(("Predict steps: {} ({})".format(None, "computed (all predict instances)")))
    tf.compat.v1.logging.info("===========================")

    predict_input_fn = get_predict_input_fn()

    estimator = get_estimator(run_config)

    predictions = estimator.predict(input_fn=predict_input_fn)

    for instance_prediction in predictions:
        tf.compat.v1.logging.info(str(instance_prediction))

    tf.compat.v1.logging.info("Done.")


def run_deep_recsys(args):
    # ******************************************************************************
    # THIS IS ENTRY POINT FOR THE TRAINER TASK
    # ******************************************************************************

    # Set python level verbosity
    tf.compat.v1.logging.set_verbosity(HYPER_PARAMS.verbosity)

    tf.compat.v1.logging.info('---------------------')
    tf.compat.v1.logging.info('Hyper-parameters:')
    tf.compat.v1.logging.info(HYPER_PARAMS)
    tf.compat.v1.logging.info('---------------------')


    # Set C++ Graph Execution level verbosity
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.compat.v1.logging.__dict__[HYPER_PARAMS.verbosity] / 10)

    # Directory to store output model and checkpoints
    model_dir = HYPER_PARAMS.job_dir
    metadata.HYPER_PARAMS = HYPER_PARAMS

    # compute the number of training steps based on num_epoch, train_size, and train_batch_size
    if HYPER_PARAMS.train_size is not None and HYPER_PARAMS.num_epochs is not None:
        HYPER_PARAMS.train_steps = int( (HYPER_PARAMS.train_size / HYPER_PARAMS.train_batch_size) * \
                      HYPER_PARAMS.num_epochs )

    num_steps_for_checkpoint = int(HYPER_PARAMS.train_steps / HYPER_PARAMS.num_checkpoints)

    run_config = tf.estimator.RunConfig(
        #tf_random_seed=19830610,  # TODO: move to hyperparameters
        save_checkpoints_steps=num_steps_for_checkpoint,  # TODO: move to hyperparameters
        log_step_count_steps=1,
        # save_checkpoints_secs=120,  #TODO: param to change if you want to change frequency of saving checkpoints
        #keep_checkpoint_max=3,
        model_dir=model_dir,
        save_summary_steps=1,
    )

    run_config = run_config.replace(model_dir=model_dir)
    tf.compat.v1.logging.info(("Model Directory:", run_config.model_dir))

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    tf.compat.v1.logging.info("")
    tf.compat.v1.logging.info(("Experiment started at {}".format(time_start.strftime("%H:%M:%S"))))
    tf.compat.v1.logging.info(".......................................")

    if HYPER_PARAMS.train:
        clean_job_dir()
        train_model(run_config)
    if HYPER_PARAMS.test:
        test_model(run_config)
    if HYPER_PARAMS.predict:
        predict_instances(run_config)

    time_end = datetime.utcnow()
    tf.compat.v1.logging.info(".......................................")
    tf.compat.v1.logging.info(("Experiment finished at {}".format(time_end.strftime("%H:%M:%S"))))
    tf.compat.v1.logging.info("")
    time_elapsed = time_end - time_start
    tf.compat.v1.logging.info(("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds())))
    tf.compat.v1.logging.info("")


def run_baseline_recsys(args):
    data_handler = DataHandler()
    predictor = get_model(args)
    train, valid, test = data_handler.load_data(args.data_dir)
    predictor.train(train)
    y_pred = predictor.predict(test)
    y_true = test.label.values
    metrics = RecallEvaluator.evaluate(y_true, y_pred)
    print(metrics)
    fname = os.path.join(args.job_dir, 'predictions.csv'.format(args.estimator))
    with tf.io.gfile.GFile(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(y_pred)
    print('done')


def run(args):
    prepare_dataset(args)

    #prepare_jobdir
    if not args.job_dir.startswith('gs://') and not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir, exist_ok=True)

    if args.estimator in [ModelName.RANDOM, ModelName.TFIDF]:
        run_baseline_recsys(args)
    else:
        run_deep_recsys(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234) # for reproducibility
    parser.add_argument('--job-dir', required=True, type=lambda x: os.path.expanduser(x))
    parser.add_argument('--data-dir', required=True, type=lambda x: os.path.expanduser(x), help='it can be a local/gcloud path or name')
    initialise_params(parser)
    HYPER_PARAMS = parser.parse_args()
    run(HYPER_PARAMS)
