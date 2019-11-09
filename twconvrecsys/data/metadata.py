

import tensorflow as tf

HYPER_PARAMS = None




def get_text_feature_size():
	# TEXT_FEATURE_SIZE = 160
	return HYPER_PARAMS.max_content_len


COL_SOURCE='source'
COL_SOURCE_LEN='source_len'
COL_TARGET='target'
COL_TARGET_LEN='target_len'
COL_LABEL = 'label'


# A List of all the columns (header) present in the input datasets file(s) in order to parse it.
# Note that, not all the columns present here will be input features to your model.
HEADER = [COL_SOURCE, COL_TARGET, COL_SOURCE_LEN, COL_TARGET_LEN, 'label']

# List of the default values of all the columns present in the input datasets.
# This helps decoding the datasets types of the columns.
HEADER_DEFAULTS = []

# List of the feature names of type int or float.
INPUT_NUMERIC_FEATURE_NAMES = [COL_SOURCE, COL_TARGET, COL_SOURCE_LEN, COL_TARGET_LEN]


def get_input_numeric_features():
	INPUT_NUMERIC_FEATURES = [('source', get_text_feature_size(), tf.int64), ('source_len', 1, tf.int64),
	                          ('target', get_text_feature_size(), tf.int64), ('target_len', 1, tf.int64)]
	return INPUT_NUMERIC_FEATURES


# Numeric features constructed, if any, in process_features function in input.py module,
# as part of reading datasets.
CONSTRUCTED_NUMERIC_FEATURE_NAMES = []

# Dictionary of feature names with int values, but to be treated as categorical features.
# In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values).
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# Categorical features with identity constructed, if any, in process_features function in input.py module,
# as part of reading datasets. Usually include constructed boolean flags.
CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# Dictionary of categorical features with few nominal values (to be encoded as one-hot indicators).
# In the dictionary, the key is the feature name, and the value is the list of feature vocabulary.
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {}

# Dictionary of categorical features with many values (sparse features).
# In the dictionary, the key is the feature name, and the value is the bucket size.
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# List of all the categorical feature names.
# This is programmatically created based on the previous inputs.
INPUT_CATEGORICAL_FEATURE_NAMES = list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

# List of all the input feature names to be used in the model.
# This is programmatically created based on the previous inputs.
INPUT_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + INPUT_CATEGORICAL_FEATURE_NAMES


def get_input_features():
	INPUT_FEATURES = get_input_numeric_features()
	return INPUT_FEATURES


# Column includes the relative weight of each record.
WEIGHT_COLUMN_NAME = None

# Target feature name (response or class variable).
TARGET_FEATURE = ('label', 1, tf.int64)

# List of the class values (labels) in a classification datasets.
TARGET_LABELS = [0, 1]

# List of the columns expected during serving (which is probably different to the header of the training datasets).
SERVING_COLUMNS = []

# List of the default values of all the columns of the serving datasets.
# This helps decoding the datasets types of the columns.
SERVING_DEFAULTS = []
