import os
import csv
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from twconvrecusers.datasets import input, metadata
from twconvrecusers.datasets.csvreader import DataHandler
from twconvrecusers.metrics.recall import RecallEvaluator
from twconvrecusers.models import neural
from twconvrecusers.models import mf
from twconvrecusers.models.factory import get_model


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

    predictions = estimator.predict(input_fn=test_input_fn)
    predictions_probs = []
    num_instances_recall = HYPER_PARAMS.num_distractors + 1

    count = 0
    path = os.path.join(HYPER_PARAMS.job_dir, 'predictions.csv')
    with open(path, 'w') as f:
        csvwriter = csv.writer(f)
        for instance_prediction in tqdm(predictions):
            #tf.compat.v1.logging.info(str(instance_prediction))
            predictions_probs.append(instance_prediction['logistic'][0])
            count += 1
            if count % num_instances_recall == 0:
                csvwriter.writerow(predictions_probs)
                predictions_probs = []

    # predictions_probs= np.split(predictions_probs, num_instances_recall, 0)
    # predictions_probs = np.concatenate(predictions_probs, 1)


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

    # fill paths based on datasets directory
    if args.data_dir:
        args.train_files = os.path.join(args.data_dir, args.train_files)
        args.eval_files = os.path.join(args.data_dir, args.eval_files)
        args.test_files = os.path.join(args.data_dir, args.test_files)
        args.vocab_path = os.path.join(args.data_dir, args.vocab_path)
        #args.vocab_proc = os.path.join(args.data_dir, 'vocab_processor.bin')
        if args.embedding_path:
            args.embedding_path = os.path.join(args.data_dir, args.embedding_path)

    #input.set_hyperparams(HYPER_PARAMS)

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
        tf_random_seed=19830610,
        save_checkpoints_steps=num_steps_for_checkpoint,  # TODO: allow to config this parameters
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


def initialise_hyper_params(args_parser):
    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the HYPER_PARAMS global variable

    Args:
        args_parser
    """

    # Data files arguments
    args_parser.add_argument(
        '--train-files',
        help='GCS or local paths to training datasets, if data-dir is provided, the path is concatenated',
        #nargs='+',
        # required=True,
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--eval-files',
        help='GCS or local paths to metrics datasets, if data-dir is provided, the path is concatenated',
        #nargs='+',
        # required=True,
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--test-files',
        help='GCS or local paths to tests datasets, if data-dir is provided, the path is concatenated',
        #nargs='+',
        # required=True,
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--predict-files',
        help='GCS or local paths to predict datasets, if data-dir is provided, the path is concatenated',
        #nargs='+',
        # required=True,
        type=lambda x: os.path.expanduser(x)
    )
    # args_parser.add_argument(
    #     '--feature-stats-file',
    #     help='GCS or local paths to feature statistics json file',
    #     #nargs='+',
    #     default=None
    # )
    args_parser.add_argument(
        '--file-encoding',
        help='file encoding',
        choices=['csv', 'tf'],
        default='tf'
    )

    args_parser.add_argument(
        '--embedding-path',
        help='Path to embeddings file, if data-dir is provided, the path is concatenated',
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--vocab-path',
        help='Path to vocabulary file, if data-dir is provided, the path is concatenated',
        type=lambda x: os.path.expanduser(x)
    )

    ###########################################

    # Experiment arguments - training
    args_parser.add_argument(
        '--train-steps',
        help="""
        Steps to run the training job for. If --num-epochs and --train-size are not specified,
        this must be. Otherwise the training job will run indefinitely.
        if --num-epochs and --train-size are specified, then --train-steps will be:
        (train-size/train-batch-size) * num-epochs\
        """,
        default=1000,
        type=int
    )
    args_parser.add_argument(
        '--train-batch-size',
        help='Batch size for each training step',
        type=int,
        default=200
    )
    args_parser.add_argument(
        '--train-size',
        help='Size of training set (instance count)',
        type=int,
        default=None
    )
    args_parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training datasets epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps will be: (train-size/train-batch-size) * num-epochs.\
        """,
        default=10,
        type=int,
    )
    args_parser.add_argument(
        '--num-checkpoints',
        default=20,
        type=int,
    )
    ###########################################

    # Experiment arguments - metrics
    args_parser.add_argument(
        '--eval-every-secs',
        help="""\
        How long to wait before running the next metrics,
        default value of 1 will eval each checkpoint (only when new checkpoint is available)
        """,
        default=1,
        type=int
    )
    args_parser.add_argument(
        '--eval-steps',
        help="""\
        Number of steps to run metrics for at each checkpoint',
        Set to None to evaluate on the whole evaluation or test set
        """,
        default=None,
        type=int
    )
    args_parser.add_argument(
        '--eval-batch-size',
        help='Batch size for metrics steps',
        type=int,
        default=200
    )
    args_parser.add_argument(
        '--num-distractors',
        help='Number of distractors in metrics datasets',
        type=int,
        default=9
    )

    args_parser.add_argument(
        '--predict-batch-size',
        help='Batch size for each prediction step',
        type=int,
        default=100
    )
    ###########################################

    # Features processing arguments

    args_parser.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns. value of 0 means no embedding',
        default=50,
        type=int
    )
    args_parser.add_argument(
        '--embedding-trainable',
        help="""\
            If set to True, the embeddings will be fine tuned during training
            """,
        action='store_true',
        #default=True,
    )
    args_parser.add_argument(
        '--vocab-size',
        help='Max number of features to use (-1: use all)',  # todo: pending implementation
        default=-1,
        type=int
    )

    ###########################################

    # Estimator arguments

    # args_parser.add_argument(
    #     '--estimator',
    #     help="Learning rate value for the optimizers",
    #     choices=[model.MODEL_RNN, model.MODEL_LSTM, model.MODEL_BiLSTM],
    #     default=model.MODEL_LSTM,
    #     type=str,
    # )

    args_parser.add_argument(
        '--learning-rate',
        help="Learning rate value for the optimizers",
        default=0.1,
        type=float
    )
    args_parser.add_argument(
        '--learning-rate-decay-factor',
        help="""\
             **VALID FOR CUSTOM MODELS**
             The factor by which the learning rate should decay by the end of the training.
             decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
             If set to 1.0 (default), then no decay will occur
             If set to 0.5, then the learning rate should reach 0.5 of its original value at the end of the training. 
             Note that, decay_steps is set to train_steps\
             """,
        default=1.0,
        type=float
    )
    args_parser.add_argument(
        '--hidden-units',
        help="""\
             Hidden layer sizes to use for DNN feature columns, provided in comma-separated layers. 
             If --scale-factor > 0, then only the size of the first layer will be used to compute 
             the sizes of subsequent layers \
             """,
        default='30,30,30'
    )
    args_parser.add_argument(
        '--layer-sizes-scale-factor',
        help="""\
            Determine how the size of the layers in the DNN decays. 
            If value = 0 then the provided --hidden-units will be taken as is\
            """,
        default=0.7,
        type=float
    )
    args_parser.add_argument(
        '--num-layers',
        help='Number of layers in the DNN. If --scale-factor > 0, then this parameter is ignored',
        default=4,
        type=int
    )
    args_parser.add_argument(
        '--dropout-prob',
        help="The probability we will drop out a given coordinate",
        default=None
    )
    args_parser.add_argument(
        '--encode-one-hot',
        help="""\
        If set to True, the categorical columns will be encoded as One-Hot indicators in the deep part of the DNN model.
        Otherwise, the categorical columns will only be used in the wide part of the DNN model
        """,
        action='store_true',
        default=True,
    )
    args_parser.add_argument(
        '--as-wide-columns',
        help="""\
        If set to True, the categorical columns will be used in the wide part of the DNN model
        """,
        action='store_true',
        default=True,
    )
    ###########################################

    # Sequence model hyperparameters

    args_parser.add_argument(
        '--rnn-dim',
        help='Dimensionality of the RNN cell',
        default=50,
        type=int
    )

    args_parser.add_argument(
        '--max-content-len',
        help='Truncate conversations contexts to this length',
        default=160,
        type=int
    )

    args_parser.add_argument(
        '--max-utterance-len',
        help='Truncate users utterance to this length',
        default=160,
        type=int
    )

    ###########################################

    # Saved model arguments
    # args_parser.add_argument(
    #     '--job-dir',
    #     help='GCS location to write checkpoints and export models',
    #     required=True,
    #     type=lambda x: os.path.expanduser(x)
    # )
    args_parser.add_argument(
        '--reuse-job-dir',
        action='store_true',
        default=False,
        help="""\
            Flag to decide if the model checkpoint should
            be re-used from the job-dir. If False then the
            job-dir will be deleted"""
    )
    args_parser.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='EXAMPLE'
    )
    ###########################################

    # Argument to turn on all logging
    args_parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )
    # Argument to turn on tfdbg debugging
    args_parser.add_argument(
        '--debug',
        action='store_true',
        help='allow to use tfdbg'
    )
    args_parser.add_argument(
        '--train',
        action='store_true',
        help='training'
    )
    args_parser.add_argument(
        '--test',
        action='store_true',
        help='testing'
    )
    args_parser.add_argument(
        '--predict',
        action='store_true',
        help='prediction'
    )

    # return args_parser.parse_args()


def run_baseline_recsys(args):
    data_handler = DataHandler()
    predictor = get_model(args)
    train, valid, test = data_handler.load_data(args.data_dir)
    predictor.train(train)
    y_pred = [predictor.predict(row['context'], row[1:]) for ix, row in test.iterrows()]
    y_pred = np.array(y_pred)
    y_true = np.zeros(test.shape[0])
    metrics = RecallEvaluator.evaluate(y_true, y_pred)
    print(metrics)
    # save predictions
    fname = os.path.join(args.job_dir, f'results_{args.estimator}.csv')
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(y_pred)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234) # for reproducibility
    parser.add_argument('--job-dir', required=True, type=lambda x: os.path.expanduser(x))
    parser.add_argument('--data-dir', type=lambda x: os.path.expanduser(x))
    #parser.add_argument('--estimator', choices=['random', 'tfidf'])
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('random')
    subparser.add_argument('--estimator', default='random')
    subparser.set_defaults(func=run_baseline_recsys)

    subparser = subparsers.add_parser('tfidf')
    subparser.add_argument('--estimator', default='tfidf')
    subparser.set_defaults(func=run_baseline_recsys)

    subparser = subparsers.add_parser('rnn')
    initialise_hyper_params(subparser)
    subparser.add_argument('--estimator', default='rnn')
    subparser.set_defaults(func=run_deep_recsys)

    subparser = subparsers.add_parser('lstm')
    initialise_hyper_params(subparser)
    subparser.add_argument('--estimator', default='lstm')
    subparser.set_defaults(func=run_deep_recsys)

    subparser = subparsers.add_parser('bilstm')
    initialise_hyper_params(subparser)
    subparser.add_argument('--estimator', default='bilstm')
    subparser.set_defaults(func=run_deep_recsys)

    subparser = subparsers.add_parser('mf')
    initialise_hyper_params(subparser)
    subparser.add_argument('--estimator', default='mf')
    subparser.set_defaults(func=run_deep_recsys)


    HYPER_PARAMS = parser.parse_args()
    HYPER_PARAMS.func(HYPER_PARAMS)
    # main()
