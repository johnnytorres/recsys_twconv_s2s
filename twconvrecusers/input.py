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


import json
import multiprocessing

import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow import data

from twconvrecusers import metadata
from twconvrecusers import featurizer
from twconvrecusers import task


# **************************************************************************
# YOU NEED NOT TO CHANGE THESE FUNCTIONS TO PARSE THE INPUT RECORDS
# **************************************************************************


def parse_csv(csv_row, is_serving=False):
    """Takes the string input tensor (csv) and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its data type
    (inferred from the metadata)

    Args:
        csv_row: rank-2 tensor of type string (csv)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving csv_row input is different than the training input (i.e., no target column)
    Returns:
        rank-2 tensor of the correct data type
    """

    if is_serving:
        column_names = metadata.SERVING_COLUMNS
        defaults = metadata.SERVING_DEFAULTS
    else:
        column_names = metadata.HEADER
        defaults = metadata.HEADER_DEFAULTS

    columns = tf.decode_csv(tf.expand_dims(csv_row, -1), record_defaults=defaults)
    features = dict(list(zip(column_names, columns)))

    return features


def parse_tf_example(example_proto, is_serving=False, mode=tfc.learn.ModeKeys.TRAIN):
    """Takes the string input tensor (example proto) and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its data type
    (inferred from the  metadata)

    Args:
        example_proto: rank-2 tensor of type string (example proto)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving csv_row input is different than the training input (i.e., no target column)
    Returns:
        rank-2 tensor of the correct data type
    """

    feature_spec = {}

    for feature_name, dimension, dtype in metadata.get_input_features():
        feature_spec[feature_name] = tf.FixedLenFeature(shape=dimension, dtype=dtype)

    for feature_name in metadata.INPUT_CATEGORICAL_FEATURES:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=1, dtype=tf.string)
        
    if mode == tfc.learn.ModeKeys.EVAL:
        num_distractors = task.HYPER_PARAMS.num_distractors
        for i in range(num_distractors):
            feature_name = 'distractor_{}'.format(i)
            feature_name_len = 'distractor_{}_len'.format(i)
            feature_spec[feature_name] = tf.FixedLenFeature(shape=metadata.get_text_feature_size(), dtype=tf.int64)
            feature_spec[feature_name_len] = tf.FixedLenFeature(shape=1, dtype=tf.int64)

    if not is_serving:
        if mode == tfc.learn.ModeKeys.TRAIN:
            feature_spec[metadata.TARGET_FEATURE[0]] = tf.FixedLenFeature(shape=metadata.TARGET_FEATURE[1], dtype=metadata.TARGET_FEATURE[2])

    parsed_features = tf.parse_example(serialized=[example_proto], features=feature_spec)

    return parsed_features


# **************************************************************************
# YOU MAY IMPLEMENT THIS FUNCTION FOR CUSTOM FEATURE ENGINEERING
# **************************************************************************


def process_features(features, mode = tfc.learn.ModeKeys.TRAIN):
    """ Use to implement custom feature engineering logic, e.g. polynomial expansion, etc.

    Default behaviour is to return the original feature tensors dictionary as-is.

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
        {string:tensors}: extended feature tensors dictionary
    """

    # examples - given:
    # 'x' and 'y' are two numeric features:
    # 'alpha' and 'beta' are two categorical features
    features['context']= tf.squeeze(features['context'], squeeze_dims=[0])
    features['context_len'] = tf.squeeze(features['context_len'], squeeze_dims=[0])
    features['utterance'] = tf.squeeze(features['utterance'], squeeze_dims=[0])
    features['utterance_len'] = tf.squeeze(features['utterance_len'], squeeze_dims=[0])

    if mode == tfc.learn.ModeKeys.EVAL:
        num_distractors = task.HYPER_PARAMS.num_distractors
        
        for i in range(num_distractors):
            feature_name = 'distractor_{}'.format(i)
            feature_name_len = 'distractor_{}_len'.format(i)
            features[feature_name] = tf.squeeze(features[feature_name], squeeze_dims=[0])
            features[feature_name_len] = tf.squeeze(features[feature_name_len], squeeze_dims=[0])


        
        

    # create new features using custom logic
    # features['x_2'] = tf.pow(features['x'],2)
    # features['y_2'] = tf.pow(features['y'], 2)
    # features['xy'] = features['x'] * features['y']
    # features['sin_x'] = tf.sin(features['x'])
    # features['cos_y'] = tf.cos(features['x'])
    # features['log_xy'] = tf.log(features['xy'])
    # features['sqrt_xy'] = tf.sqrt(features['xy'])

    # create boolean flags
    # features['x_grt_y'] = tf.cast(features['x'] > features['y'], tf.int32)
    # features['alpha_eq_beta'] = features['alpha'] == features['beta']

    # add created features to metadata (if not already defined in metadata.py)
    # CONSTRUCTED_NUMERIC_FEATURE_NAMES += ['x_2', 'y_2', 'xy', ....]
    # CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY['x_grt_y'] = 2
    # CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY['alpha_eq_beta'] = 2

    return features


def get_features_target_tuple(features, mode=tfc.learn.ModeKeys.TRAIN):
    """ Get a tuple of input feature tensors and target feature tensor.

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
          {string:tensors}, {tensor} -  input feature tensor dictionary and target feature tensor
    """

    unused_features = list(set(metadata.HEADER) -
                           set(metadata.INPUT_FEATURE_NAMES) -
                           {metadata.TARGET_NAME} -
                           {metadata.WEIGHT_COLUMN_NAME})

    # remove unused columns (if any)
    for column in unused_features:
        features.pop(column, None)
        
    if mode == tfc.learn.ModeKeys.EVAL:
        features[metadata.TARGET_NAME] = tf.zeros((1,1), dtype=tf.int64, name=metadata.TARGET_NAME)

    # get target feature
    target = features.pop(metadata.TARGET_NAME)
    target = tf.squeeze(target, squeeze_dims=[0])


    return features, target


def posprocessing(features, target, mode):
    if mode == tfc.learn.ModeKeys.EVAL:
        num_distractors = task.HYPER_PARAMS.num_distractors
        
        context = features['context']
        context_len = features['context_len']
        utterance = features['utterance']
        utterance_len = features['utterance_len']
        all_context = [context]
        all_context_len = [context_len]
        all_utterances = [utterance]
        all_utterances_len = [utterance_len]
        all_targets = [tf.ones((tf.shape(context)[0], 1), dtype=tf.int64)]
        
        for i in range(num_distractors):
            feature_name = 'distractor_{}'.format(i)
            feature_name_len = 'distractor_{}_len'.format(i)
            distractor = features[feature_name]
            distractor_len = features[feature_name_len]
            all_context.append(context)
            all_context_len.append(context_len)
            all_utterances.append(distractor)
            all_utterances_len.append(distractor_len)
            all_targets.append(tf.zeros((tf.shape(context)[0], 1), dtype=tf.int64))
        
        all_context = tf.concat(all_context, 0)
        all_context_len = tf.concat(all_context_len, 0)
        all_utterances = tf.concat(all_utterances, 0)
        all_utterances_len = tf.concat(all_utterances_len, 0)
        all_targets = tf.concat(all_targets, 0)

        # all_targets = tf.tf.logging.info(all_targets, [all_targets], summarize=100)
        
        features = {}
        features['context'] = all_context
        features['context_len'] = all_context_len
        features['utterance'] = all_utterances
        features['utterance_len'] = all_utterances_len
        target = all_targets
    
    return features, target


# **************************************************************************
# YOU NEED NOT TO CHANGE THIS FUNCTION TO READ DATA FILES
# **************************************************************************




def generate_input_fn(file_names_pattern,
                     file_encoding='csv',
                     mode=tf.estimator.ModeKeys.EVAL,
                     skip_header_lines=0,
                     num_epochs=1,
                     batch_size=200,
                     multi_threading=True):
    """Generates an input function for reading training and evaluation data file(s).
    This uses the tf.data APIs.

    Args:
        file_names_pattern: [str] - file name or file name patterns from which to read the data.
        mode: tf.estimator.ModeKeys - either TRAIN or EVAL.
            Used to determine whether or not to randomize the order of data.
        file_encoding: type of the text files. Can be 'csv' or 'tfrecords'
        skip_header_lines: int set to non-zero in order to skip header lines in CSV files.
        num_epochs: int - how many times through to read the data.
          If None will loop through data indefinitely
        batch_size: int - first dimension size of the Tensors returned by input_fn
        multi_threading: boolean - indicator to use multi-threading or not
    Returns:
        A function () -> (features, indices) where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """
    def _input_fn():

        shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

        data_size = task.HYPER_PARAMS.train_size if mode == tf.estimator.ModeKeys.TRAIN else None

        num_threads = multiprocessing.cpu_count() if multi_threading else 1

        buffer_size = 2 * batch_size + 1

        tf.logging.info("")
        tf.logging.info("* data input_fn:")
        tf.logging.info("================")
        tf.logging.info(("Mode: {}".format(mode)))
        tf.logging.info(("Input file(s): {}".format(file_names_pattern)))
        tf.logging.info(("Files encoding: {}".format(file_encoding)))
        tf.logging.info(("Data size: {}".format(data_size)))
        tf.logging.info(("Batch size: {}".format(batch_size)))
        tf.logging.info(("Epoch count: {}".format(num_epochs)))
        tf.logging.info(("Thread count: {}".format(num_threads)))
        tf.logging.info(("Shuffle: {}".format(shuffle)))
        tf.logging.info("================")
        tf.logging.info("")

        file_names = tf.matching_files(file_names_pattern)

        if file_encoding == 'csv':
            dataset = data.TextLineDataset(filenames=file_names)
            dataset = dataset.skip(skip_header_lines)
            dataset = dataset.map(lambda csv_row: parse_csv(csv_row))

        else:
            dataset = data.TFRecordDataset(filenames=file_names)
            dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, mode=mode),
                                  num_parallel_calls=num_threads)

        dataset = dataset.map(lambda features: get_features_target_tuple(features, mode=mode),
                              num_parallel_calls=num_threads)
        dataset = dataset.map(lambda features, target: (process_features(features, mode=mode), target),
                              num_parallel_calls=num_threads)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size)
        dataset = dataset.repeat(num_epochs)

        iterator = dataset.make_one_shot_iterator()
        features, target = iterator.get_next()

        features, target = posprocessing(features, target, mode)

        return features, target

    return _input_fn


# **************************************************************************
# YOU MAY CHANGE THIS FUNCTION TO LOAD YOUR NUMERIC COLUMN STATS
# **************************************************************************


def load_feature_stats():
    """
    Load numeric column pre-computed statistics (mean, stdv, min, max, etc.)
    in order to be used for scaling/stretching numeric columns.

    In practice, the statistics of large datasets are computed prior to model training,
    using dataflow (beam), dataproc (spark), BigQuery, etc.

    The stats are then saved to gcs location. The location is passed to package
    in the --feature-stats-file argument. However, it can be a local path as well.

    Returns:
        json object with the following schema: stats['feature_name']['state_name']
    """

    feature_stats = None
    try:
        if task.HYPER_PARAMS.feature_stats_file is not None and tf.gfile.Exists(task.HYPER_PARAMS.feature_stats_file):
            with tf.gfile.Open(task.HYPER_PARAMS.feature_stats_file) as file:
                content = file.read()
            feature_stats = json.loads(content)
            tf.logging.info("feature stats were successfully loaded from local file...")
        else:
            tf.logging.warn("feature stats file not found. numerical columns will not be normalised...")
    except:
        tf.logging.warn("couldn't load feature stats. numerical columns will not be normalised...")

    return feature_stats


# ****************************************************************************
# SERVING FUNCTIONS - YOU NEED NOT TO CHANGE THE FOLLOWING PART
# ****************************************************************************


def json_serving_input_fn():
    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

    inputs = {}
    
    # for feature_name, dimension, dtype in metadata.INPUT_FEATURES:
    #     feature_spec[feature_name] = tf.FixedLenFeature(shape=dimension, dtype=dtype)

    for column in input_feature_columns:
        if column.name in metadata.INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY:
            inputs[column.name] = tf.placeholder(shape=[None], dtype=tf.int32)
        else:
            inputs[column.name] = tf.placeholder(shape=column.shape, dtype=column.dtype)

    features = {
        key: tf.expand_dims(tensor, 0)
        for key, tensor in list(inputs.items())
    }

    return tf.estimator.export.ServingInputReceiver(
        features=process_features(features),
        receiver_tensors=inputs
    )


def csv_serving_input_fn():
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )

    features = parse_csv(csv_row, is_serving=True)

    unused_features = list(set(metadata.SERVING_COLUMNS) - set(metadata.INPUT_FEATURE_NAMES) - {metadata.TARGET_NAME})

    # Remove unused columns (if any)
    for column in unused_features:
        features.pop(column, None)

    return tf.estimator.export.ServingInputReceiver(
        features=process_features(features),
        receiver_tensors={'csv_row': csv_row}
    )


def example_serving_input_fn():
    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    
    feature_scalars = tf.parse_example(
        example_bytestring,
        tf.feature_column.make_parse_example_spec(input_feature_columns)
    )

    # features = {
    #     key: tf.expand_dims(tensor, -1)
    #     for key, tensor in feature_scalars.items()
    # }
    
    features = feature_scalars

    return tf.estimator.export.ServingInputReceiver(
        features=process_features(features),
        receiver_tensors={'example_proto': example_bytestring}
    )


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}

