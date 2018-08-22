#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from trainer import input
from trainer import metadata
from trainer import model


# ******************************************************************************
# YOU MAY MODIFY THIS FUNCTION TO ADD/REMOVE PARAMS OR CHANGE THE DEFAULT VALUES
# ******************************************************************************


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
        help='GCS or local paths to training data',
        nargs='+',
        #required=True,
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--eval-files',
        help='GCS or local paths to evaluation data',
        nargs='+',
        #required=True,
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--test-files',
        help='GCS or local paths to test data',
        nargs='+',
        #required=True,
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--predict-files',
        help='GCS or local paths to predict data',
        nargs='+',
        #required=True,
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--feature-stats-file',
        help='GCS or local paths to feature statistics json file',
        nargs='+',
        default=None
    )
    args_parser.add_argument(
        '--file-encoding',
        help='file encoding',
        choices=['csv','tf'],
        default='csv'
    )
    
    args_parser.add_argument(
        '--embedding-path',
        help='Path to embeddings file',
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--vocab-path',
        help='Path to vocabulary file',
        type=lambda x: os.path.expanduser(x)
    )
    args_parser.add_argument(
        '--vocab-proc',
        help='Path to vocabulary preprocessor file',
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
        Maximum number of training data epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps will be: (train-size/train-batch-size) * num-epochs.\
        """,
        default=10,
        type=int,
    )
    ###########################################

    # Experiment arguments - evaluation
    args_parser.add_argument(
        '--eval-every-secs',
        help='How long to wait before running the next evaluation',
        default=120,
        type=int
    )
    args_parser.add_argument(
        '--eval-steps',
        help="""\
        Number of steps to run evaluation for at each checkpoint',
        Set to None to evaluate on the whole evaluation data
        """,
        default=None,
        type=int
    )
    args_parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=200
    )
    args_parser.add_argument(
        '--num-distractors',
        help='Number of distractors in evaluation dataset',
        type=int,
        default=9
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
        default=False,
    )
    args_parser.add_argument(
        '--vocab-size',
        help='Max number of features to use (-1: use all)', #todo: pending implementation
        default=-1,
        type=int
    )


    ###########################################

    # Estimator arguments
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
        default=256,
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
    args_parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True,
        type=lambda x: os.path.expanduser(x)
    )
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
    

    return args_parser.parse_args()


# ******************************************************************************
# YOU NEED NOT TO CHANGE THE FUNCTION TO RUN THE EXPERIMENT
# ******************************************************************************

def clean_job_dir():
    # If job_dir_reuse is False then remove the job_dir if it exists
    tf.logging.info(("Resume training:", HYPER_PARAMS.reuse_job_dir))
    if not HYPER_PARAMS.reuse_job_dir:
        if tf.gfile.Exists(HYPER_PARAMS.job_dir):
            tf.gfile.DeleteRecursively(HYPER_PARAMS.job_dir)
            tf.logging.info(("Deleted job_dir {} to avoid re-use".format(HYPER_PARAMS.job_dir)))
        else:
            tf.logging.info("No job_dir available to delete")
    else:
        tf.logging.info(("Reusing job_dir {} if it exists".format(HYPER_PARAMS.job_dir)))
        
        
def get_train_input_fn():
    train_input_fn = input.generate_input_fn(
        file_names_pattern=HYPER_PARAMS.train_files,
        file_encoding=HYPER_PARAMS.file_encoding,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_epochs=HYPER_PARAMS.num_epochs,
        batch_size=HYPER_PARAMS.train_batch_size
    )
    return train_input_fn


def get_eval_input_fn():
    eval_input_fn = input.generate_input_fn(
        file_names_pattern=HYPER_PARAMS.eval_files,
        file_encoding=HYPER_PARAMS.file_encoding,
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=HYPER_PARAMS.eval_batch_size
    )
    return eval_input_fn


def get_test_input_fn():
    test_input_fn = input.generate_input_fn(
        file_names_pattern=HYPER_PARAMS.test_files,
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
        batch_size=HYPER_PARAMS.train_batch_size
    )
    return predict_input_fn


def train_model(run_config):
    """Train, evaluate, and export the model using tf.estimator.train_and_evaluate API"""

    train_input_fn = get_train_input_fn()

    eval_input_fn = get_eval_input_fn()

    exporter = tf.estimator.FinalExporter(
        'estimator',
        input.SERVING_FUNCTIONS[HYPER_PARAMS.export_format],
        as_text=False  # change to true if you want to export the model as readable text
    )

    # compute the number of training steps based on num_epoch, train_size, and train_batch_size
    if HYPER_PARAMS.train_size is not None and HYPER_PARAMS.num_epochs is not None:
        train_steps = (HYPER_PARAMS.train_size / HYPER_PARAMS.train_batch_size) * \
                      HYPER_PARAMS.num_epochs
    else:
        train_steps = HYPER_PARAMS.train_steps

    hooks = []
    if HYPER_PARAMS.debug:
        hooks =[tf_debug.LocalCLIDebugHook()]

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=int(train_steps),
        hooks = hooks
    )

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=HYPER_PARAMS.eval_steps,
        exporters=[exporter],
        name='estimator-eval',
        throttle_secs=HYPER_PARAMS.eval_every_secs,
        hooks=hooks
    )

    tf.logging.info("===========================")
    tf.logging.info("* TRAINING configurations")
    tf.logging.info("===========================")
    tf.logging.info(("Train size: {}".format(HYPER_PARAMS.train_size)))
    tf.logging.info(("Epoch count: {}".format(HYPER_PARAMS.num_epochs)))
    tf.logging.info(("Train batch size: {}".format(HYPER_PARAMS.train_batch_size)))
    tf.logging.info(("Training steps: {} ({})".format(int(train_steps),
                                           "supplied" if HYPER_PARAMS.train_size is None else "computed")))
    tf.logging.info(("Evaluate every {} seconds".format(HYPER_PARAMS.eval_every_secs)))
    tf.logging.info("===========================")

    if metadata.TASK_TYPE == "classification":
        estimator = model.create_classifier(
            config=run_config
        )
    elif metadata.TASK_TYPE == "regression":
        estimator = model.create_regressor(
            config=run_config
        )
    else:
        estimator = model.create_estimator(
            config=run_config
        )

    # train and evaluate
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )

# EVALUATE MODEL WITH TEST DATA


def test_model(run_config):
    test_input_fn = get_test_input_fn()
    tf.logging.info("===========================")
    tf.logging.info("* TESTING configurations")
    tf.logging.info("===========================")
    tf.logging.info(("Test batch size: {}".format(HYPER_PARAMS.eval_batch_size)))
    tf.logging.info(("Test steps: {} ({})".format(None, "computed (all test instances)")))
    tf.logging.info("===========================")

    estimator = model.create_estimator(
        config=run_config
    )

    hooks = []
    if HYPER_PARAMS.debug:
        hooks = [tf_debug.LocalCLIDebugHook()]

    estimator.evaluate(
        input_fn=test_input_fn,
        hooks=hooks
    )
    
# PREDICT EXAMPLE INSTANCES

    
def predict_instances(run_config):
    tf.logging.info("===========================")
    tf.logging.info("* PREDICT configurations")
    tf.logging.info("===========================")
    tf.logging.info(("Predict batch size: {}".format(HYPER_PARAMS.train_batch_size)))
    tf.logging.info(("Predict steps: {} ({})".format(None, "computed (all predict instances)")))
    tf.logging.info("===========================")

    predict_input_fn= get_predict_input_fn()
    
    estimator = model.create_estimator(
        config=run_config
    )

    predictions = estimator.predict(input_fn=predict_input_fn)
    
    for instance_prediction in predictions:
        tf.logging.info(str(instance_prediction))

    tf.logging.info("Done.")

# ******************************************************************************
# THIS IS ENTRY POINT FOR THE TRAINER TASK
# ******************************************************************************


def main():

    tf.logging.info('---------------------')
    tf.logging.info('Hyper-parameters:')
    tf.logging.info(HYPER_PARAMS)
    tf.logging.info('---------------------')

    # Set python level verbosity
    tf.logging.set_verbosity(HYPER_PARAMS.verbosity)

    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__[HYPER_PARAMS.verbosity] / 10)

    # Directory to store output model and checkpoints
    model_dir = HYPER_PARAMS.job_dir
    metadata.HYPER_PARAMS = HYPER_PARAMS

    run_config = tf.estimator.RunConfig(
        tf_random_seed=19830610,
        log_step_count_steps=1000,
        save_checkpoints_secs=120,  # change if you want to change frequency of saving checkpoints
        keep_checkpoint_max=3,
        model_dir=model_dir
    )

    run_config = run_config.replace(model_dir=model_dir)
    tf.logging.info(("Model Directory:", run_config.model_dir))

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    tf.logging.info("")
    tf.logging.info(("Experiment started at {}".format(time_start.strftime("%H:%M:%S"))))
    tf.logging.info(".......................................")

    if HYPER_PARAMS.train:
        clean_job_dir()
        train_model(run_config)
    if HYPER_PARAMS.test:
        test_model(run_config)
    if HYPER_PARAMS.predict:
        predict_instances(run_config)

    time_end = datetime.utcnow()
    tf.logging.info(".......................................")
    tf.logging.info(("Experiment finished at {}".format(time_end.strftime("%H:%M:%S"))))
    tf.logging.info("")
    time_elapsed = time_end - time_start
    tf.logging.info(("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds())))
    tf.logging.info("")


args_parser = argparse.ArgumentParser()
HYPER_PARAMS = initialise_hyper_params(args_parser)

if __name__ == '__main__':
    main()
