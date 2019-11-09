import os

from twconvrecsys.models.factory import ModelName


def initialise_params(args_parser):
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

    args_parser.add_argument(
        '--estimator',
        default=ModelName.RANDOM,
        choices=[
            ModelName.RANDOM,
            ModelName.TFIDF,
            ModelName.RNN,
            ModelName.LSTM,
            ModelName.BILSTM,
            ModelName.MF,
            ModelName.NMF
        ])


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