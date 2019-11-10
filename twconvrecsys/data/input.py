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

from twconvrecsys.data import featurizer, metadata

# **************************************************************************
# YOU NEED NOT TO CHANGE THESE FUNCTIONS TO PARSE THE INPUT RECORDS
# **************************************************************************


def parse_csv(csv_row, is_serving=False):
    """Takes the string input tensor (csv) and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its datasets type
    (inferred from the metadata)

    Args:
        csv_row: rank-2 tensor of type string (csv)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving csv_row input is different than the training input (i.e., no target column)
    Returns:
        rank-2 tensor of the correct datasets type
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


def parse_tf_example(example_proto, HYPER_PARAMS, is_serving=False, mode=tfc.learn.ModeKeys.TRAIN):
    """Takes the string input tensor (example proto) and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor, with respect to its datasets type
    (inferred from the  metadata)

    Args:
        example_proto: rank-2 tensor of type string (example proto)
        is_serving: boolean to indicate whether this function is called during serving or training
        since the serving csv_row input is different than the training input (i.e., no target column)
    Returns:
        rank-2 tensor of the correct datasets type
    """

    feature_spec = {}

    if mode == tfc.learn.ModeKeys.EVAL:
        feature_spec['source'] = tf.io.FixedLenFeature(shape=metadata.get_text_feature_size(), dtype=tf.int64)
        feature_spec['source_len'] = tf.io.FixedLenFeature(shape=1, dtype=tf.int64)
        num_targets = HYPER_PARAMS.num_distractors + 1
        for i in range(num_targets):
            feature_name = 'target_{}'.format(i)
            feature_name_len = 'target_{}_len'.format(i)
            feature_spec[feature_name] = tf.io.FixedLenFeature(shape=metadata.get_text_feature_size(), dtype=tf.int64)
            feature_spec[feature_name_len] = tf.io.FixedLenFeature(shape=1, dtype=tf.int64)
        feature_spec[metadata.TARGET_FEATURE[0]] = tf.io.FixedLenFeature(shape=metadata.TARGET_FEATURE[1],dtype=metadata.TARGET_FEATURE[2])
    else:
        for feature_name, dimension, dtype in metadata.get_input_features():
            feature_spec[feature_name] = tf.io.FixedLenFeature(shape=dimension, dtype=dtype)

        if not is_serving:
            feature_spec[metadata.TARGET_FEATURE[0]] = tf.io.FixedLenFeature(shape=metadata.TARGET_FEATURE[1], dtype=metadata.TARGET_FEATURE[2])

    parsed_features = tf.io.parse_example(serialized=[example_proto], features=feature_spec)

    return parsed_features


# **************************************************************************
# YOU MAY IMPLEMENT THIS FUNCTION FOR CUSTOM FEATURE ENGINEERING
# **************************************************************************


def process_features(features, HYPER_PARAMS, mode = tfc.learn.ModeKeys.TRAIN):
    """ Use to implement custom feature engineering logic, e.g. polynomial expansion, etc.

    Default behaviour is to return the original feature tensors dictionary as-is.

    Args:
        features: {string:tensors} - dictionary of feature tensors
    Returns:
        {string:tensors}: extended feature tensors dictionary
    """

    if mode == tfc.learn.ModeKeys.EVAL:
        num_distractors = HYPER_PARAMS.num_distractors
        features['source'] = tf.squeeze(features['source'], squeeze_dims=[0])
        features['source_len'] = tf.squeeze(features['source_len'], squeeze_dims=[0])

        for i in range(num_distractors+1):
            feature_name = 'target_{}'.format(i)
            feature_name_len = 'target_{}_len'.format(i)
            features[feature_name] = tf.squeeze(features[feature_name], squeeze_dims=[0])
            features[feature_name_len] = tf.squeeze(features[feature_name_len], squeeze_dims=[0])
    else:
        features['source'] = tf.squeeze(features['source'], squeeze_dims=[0])
        features['source_len'] = tf.squeeze(features['source_len'], squeeze_dims=[0])
        features['target'] = tf.squeeze(features['target'], squeeze_dims=[0])
        features['target_len'] = tf.squeeze(features['target_len'], squeeze_dims=[0])
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
                           {metadata.COL_LABEL} -
                           {metadata.WEIGHT_COLUMN_NAME})

    # remove unused columns (if any)
    for column in unused_features:
        features.pop(column, None)
        
    # if mode == tfc.learn.ModeKeys.EVAL:
    #     features[metadata.COL_LABEL] = tf.zeros((1, 1), dtype=tf.int64, name=metadata.COL_LABEL)

    # get target feature
    target = features.pop(metadata.COL_LABEL)
    target = tf.squeeze(target, squeeze_dims=[0])


    return features, target


def posprocessing(features, labels, mode, HYPER_PARAMS):
    if mode == tfc.learn.ModeKeys.EVAL:
        num_distractors = HYPER_PARAMS.num_distractors
        
        sources = features['source']
        sources_len = features['source_len']
        all_sources = []
        all_sources_len = []
        all_targets = []
        all_targets_len = []
        all_labels = []
        n_instances = num_distractors+1 #  distractors + ground truth
        labels = tf.one_hot(labels,  depth=n_instances)
        
        for i in range(n_instances):
            all_sources.append(sources)
            all_sources_len.append(sources_len)

            feature_name = 'target_{}'.format(i)
            feature_name_len = 'target_{}_len'.format(i)
            target = features[feature_name]
            target_len = features[feature_name_len]
            all_targets.append(target)
            all_targets_len.append(target_len)

            current_labels = tf.gather(labels, i, axis=2)
            all_labels.append(current_labels)
        
        all_sources = tf.concat(all_sources, 0)
        all_sources_len = tf.concat(all_sources_len, 0)
        all_targets = tf.concat(all_targets, 0)
        all_targets_len = tf.concat(all_targets_len, 0)
        all_labels = tf.concat(all_labels, 0)

        features = {}
        features['source'] = all_sources
        features['source_len'] = all_sources_len
        features['target'] = all_targets
        features['target_len'] = all_targets_len
        labels = all_labels
    
    return features, labels


# **************************************************************************
# YOU NEED NOT TO CHANGE THIS FUNCTION TO READ DATA FILES
# **************************************************************************




def generate_input_fn(file_names_pattern,
                     HYPER_PARAMS,
                     file_encoding='csv',
                     mode=tf.estimator.ModeKeys.EVAL,
                     skip_header_lines=0,
                     num_epochs=1,
                     batch_size=200,
                     multi_threading=True):
    """Generates an input function for reading training and metrics datasets file(s).
    This uses the tf.datasets APIs.

    Args:
        file_names_pattern: [str] - file name or file name patterns from which to read the datasets.
        mode: tf.estimator.ModeKeys - either TRAIN or EVAL.
            Used to determine whether or not to randomize the order of datasets.
        file_encoding: type of the text files. Can be 'csv' or 'tfrecords'
        skip_header_lines: int set to non-zero in order to skip header lines in CSV files.
        num_epochs: int - how many times through to read the datasets.
          If None will loop through datasets indefinitely
        batch_size: int - first dimension size of the Tensors returned by input_fn
        multi_threading: boolean - indicator to use multi-threading or not
    Returns:
        A function () -> (features, indices) where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """
    def _input_fn():

        shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

        data_size = HYPER_PARAMS.train_size if mode == tf.estimator.ModeKeys.TRAIN else None

        num_threads = multiprocessing.cpu_count() if multi_threading else 1

        buffer_size = 2 * batch_size + 1

        tf.compat.v1.logging.info("")
        tf.compat.v1.logging.info("* datasets input_fn:")
        tf.compat.v1.logging.info("================")
        tf.compat.v1.logging.info(("Mode: {}".format(mode)))
        tf.compat.v1.logging.info(("Input file(s): {}".format(file_names_pattern)))
        tf.compat.v1.logging.info(("Files encoding: {}".format(file_encoding)))
        tf.compat.v1.logging.info(("Data size: {}".format(data_size)))
        tf.compat.v1.logging.info(("Batch size: {}".format(batch_size)))
        tf.compat.v1.logging.info(("Epoch count: {}".format(num_epochs)))
        tf.compat.v1.logging.info(("Thread count: {}".format(num_threads)))
        tf.compat.v1.logging.info(("Shuffle: {}".format(shuffle)))
        tf.compat.v1.logging.info("================")
        tf.compat.v1.logging.info("")

        file_names = tf.io.matching_files(file_names_pattern)

        if file_encoding == 'csv':
            dataset = data.TextLineDataset(filenames=file_names)
            dataset = dataset.skip(skip_header_lines)
            dataset = dataset.map(lambda csv_row: parse_csv(csv_row))

        else:
            dataset = data.TFRecordDataset(filenames=file_names)
            dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, HYPER_PARAMS, mode=mode),
                                  num_parallel_calls=num_threads)

        dataset = dataset.map(lambda features: get_features_target_tuple(features, mode=mode),
                              num_parallel_calls=num_threads)
        dataset = dataset.map(lambda features, target: (process_features(features, HYPER_PARAMS=HYPER_PARAMS, mode=mode), target),
                              num_parallel_calls=num_threads)

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size)
        dataset = dataset.repeat(num_epochs)

        iterator = dataset.make_one_shot_iterator()
        features, target = iterator.get_next()

        features, target = posprocessing(features, target, mode, HYPER_PARAMS)

        return features, target

    return _input_fn


# **************************************************************************
# YOU MAY CHANGE THIS FUNCTION TO LOAD YOUR NUMERIC COLUMN STATS
# **************************************************************************


def load_feature_stats(HYPER_PARAMS):
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
        if HYPER_PARAMS.feature_stats_file is not None and tf.gfile.Exists(HYPER_PARAMS.feature_stats_file):
            with tf.gfile.Open(HYPER_PARAMS.feature_stats_file) as file:
                content = file.read()
            feature_stats = json.loads(content)
            tf.compat.v1.logging.info("feature stats were successfully loaded from local file...")
        else:
            tf.compat.v1.logging.warn("feature stats file not found. numerical columns will not be normalised...")
    except:
        tf.compat.v1.logging.warn("couldn't load feature stats. numerical columns will not be normalised...")

    return feature_stats


# ****************************************************************************
# SERVING FUNCTIONS - YOU NEED NOT TO CHANGE THE FOLLOWING PART
# ****************************************************************************


def json_serving_input_fn():
    feature_columns = featurizer.create_feature_columns()
    input_feature_columns = [feature_columns[feature_name] for feature_name in metadata.INPUT_FEATURE_NAMES]

    inputs = {}
    
    # for feature_name, dimension, dtype in metadata.INPUT_FEATURES:
    #     feature_spec[feature_name] = tf.io.FixedLenFeature(shape=dimension, dtype=dtype)

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

    unused_features = list(set(metadata.SERVING_COLUMNS) - set(metadata.INPUT_FEATURE_NAMES) - {metadata.COL_LABEL})

    # Remove unused columns (if any)
    for column in unused_features:
        features.pop(column, None)

    return tf.estimator.export.ServingInputReceiver(
        features=process_features(features),
        receiver_tensors={'csv_row': csv_row}
    )




def get_serving_function(HYPER_PARAMS):
    def example_serving_input_fn():
        feature_columns = featurizer.create_feature_columns(HYPER_PARAMS)
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
            features=process_features(features, HYPER_PARAMS),
            receiver_tensors={'example_proto': example_bytestring}
        )


    SERVING_FUNCTIONS = {
        'JSON': json_serving_input_fn,
        'EXAMPLE': example_serving_input_fn,
        'CSV': csv_serving_input_fn
    }
    return SERVING_FUNCTIONS[HYPER_PARAMS.export_format]

