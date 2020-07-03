#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import datetime
import pathlib
from typing import Tuple
import tensorflow as tf
from spectrogram_sequencer import SpectrogamSequencer
from commons import AUTOTUNE
from commons import EXPERIMENTS_FOLDER
from commons import LOCAL_DIAGRAM_FOLDER
from commons import LOCAL_TRAINED_MODEL_FOLDER
from commons import verify_tfrecords_from_directory
from commons import get_classes_from_directory
from classification_utils import display_performance
import sys
sys.path.append('../extraction')
from tfrecord_dataset import feature_description

# Constants
MODELS_NAME = 'air_binary_rnn'
TIME_SIZE = 401
MFCC_SIZE = 128
WINDOW_SIZE = 50
WINDOW_OVERLAP = 0.5
BATCH_SIZE = 64
BUFFER_SIZE = 1000


# Parses observation from proto format and converts into correct format for training (input,output) = (spec,label)
@tf.function
def parse_observation(example: tf.Tensor, categories: list) -> Tuple:
  observation = tf.io.parse_single_example(example, feature_description)
  mfcc = observation['mfcc']
  samples = observation['samples']
  spec = tf.reshape(observation['spec'], (mfcc, samples))
  spec = tf.expand_dims(spec, axis=-1)  # channel
  label = observation['label'] == categories[0]
  return spec, label


# Creates dataset from tfrecord files
def create_dataset(dataset_folder: pathlib.Path, categories: list) -> Tuple:
  # Creates training data pipeline
  train_ds = tf.data.TFRecordDataset([str(dataset_folder / 'train.tfrecord')])
  train_ds = train_ds.map(lambda example: parse_observation(example, categories), num_parallel_calls=AUTOTUNE).cache()
  train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
  # Creates test data pipeline
  test_ds = tf.data.TFRecordDataset([str(dataset_folder / 'test.tfrecord')])
  test_ds = test_ds.map(lambda example: parse_observation(example, categories), num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)
  return train_ds, test_ds


class AirBinaryRNN:
  """
  Simple recurrent neural network model using a sequence of spectrogram windows. The input of this model
  will be a sequence of overlapping spectrogram slices (windows).

  Designed to perform binary classification on a simple dataset containing
  Airbus/Boeing aircraft take-off signals.
  """

  def __init__(self, dataset_folder: pathlib.Path, use_regularizer=True, use_batch_norm=False):
    # Setups experiment folder
    self.dataset_folder = dataset_folder
    self.experiment_path, self.model_path, self.diagram_path = self.setup_experiment_folder()
    # Determine classes from folder
    categories = get_classes_from_directory(dataset_folder)
    # Verify binary
    assert len(categories) == 2, 'Wrong number of classes. Expecting two.'
    self.categories = categories
    # Stores options
    self.use_regularizer = use_regularizer
    self.use_batch_norm = use_batch_norm
    # Builds model architecture
    self.model = self.build_model()
    # Selects loss and metric
    self.model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy']
    )

  # Setup experiment folder
  def setup_experiment_folder(self) -> Tuple:
    # Verify train.tfrecord and test.tfrecord exist in dataset folder
    assert verify_tfrecords_from_directory(self.dataset_folder), 'There is no train.tfrecord or test.tfrecord' \
                                                                 'in the folder specified '
    # Experiment path (root folder of the experiment)
    experiment_path = self.dataset_folder / EXPERIMENTS_FOLDER / datetime.datetime.now().strftime('%Y-%m-%d'
                                                                                                  '-%H-%M-%S')
    if not experiment_path.exists():
      experiment_path.mkdir(parents=False)
    # Model path (where it is saved during training)
    model_path = experiment_path / LOCAL_TRAINED_MODEL_FOLDER / MODELS_NAME
    if not model_path.exists():
      model_path.mkdir(parents=True)
    # Keras model diagram
    diagram_path = experiment_path / LOCAL_DIAGRAM_FOLDER / (MODELS_NAME + '.jpg')
    if not diagram_path.parent.exists():
      diagram_path.parent.mkdir(parents=False)
    # Returns all relevant paths as tuple
    return experiment_path, model_path, diagram_path

  # Builds models architecture returning a tf.keras.Model
  def build_model(self) -> tf.keras.Model:
    # Create inputs
    inputs = tf.keras.layers.Input((MFCC_SIZE, TIME_SIZE, 1))
    # Creates the sequencer layer (only used if requested)
    sequencer = SpectrogamSequencer(WINDOW_SIZE, WINDOW_OVERLAP)
    # First convolutional layer to find spatial features in a segment of the spectrogram. Timed distributed to get
    # applied to every segment of the inputted spectrogram.
    conv1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5,
                             padding='valid',
                             data_format='channels_last',
                             use_bias=False if self.use_batch_norm else True),
      name='Conv1'
    )
    # Batch norm layer (only used if requested)
    batch_norm1 = tf.keras.layers.BatchNormalization(axis=1, name='BatchNorm1')
    # Activation for convolutional layer
    activation1 = tf.keras.layers.Activation(tf.nn.relu, name='Relu1')
    # Pooling
    pooling1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3, name='Pool1')
    )
    # Second convolutional layer. Timed distributed to get applied to every segment of the inputted spectrogram.
    conv2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5,
                             padding='valid',
                             data_format='channels_last',
                             use_bias=False if self.use_batch_norm else True,
                             name='Conv2')
    )
    # Batch norm layer (only used if requested)
    batch_norm2 = tf.keras.layers.BatchNormalization(axis=1, name='BatchNorm2')
    # Activation for the second convolutional layer
    activation2 = tf.keras.layers.Activation(tf.nn.relu, name='Relu2')
    # Pooling
    pooling2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3, name='Pool2')
    )
    # Flatten results from convolutions/pooling to be fed to the lstm layers
    flatten1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Flatten(name='Flatten')
    )
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.3)
    # Recurrent layer to capture temporal relationships
    lstm1 = tf.keras.layers.LSTM(32,
                                 return_state=False,
                                 return_sequences=False,
                                 name='LSTM1')
    # Dropout layer
    dropout2 = tf.keras.layers.Dropout(0.1)
    # Dense to make the final classification
    dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.3) if self.use_regularizer else None,
                                   name='Dense')
    # Creates connections between layers of the model
    x = sequencer(inputs)
    x = conv1(x)
    if self.use_batch_norm:
      x = batch_norm1(x)
    x = activation1(x)
    x = pooling1(x)
    x = conv2(x)
    if self.use_batch_norm:
      x = batch_norm2(x)
    x = activation2(x)
    x = pooling2(x)
    x = flatten1(x)
    if self.use_regularizer:
      x = dropout1(x)
    x = lstm1(x)
    if self.use_regularizer:
      x = dropout2(x)
    outputs = dense1(x)
    return tf.keras.Model(inputs, outputs)

  # Prints out the model's summary
  def summary(self):
    self.model.summary()
    tf.keras.utils.plot_model(self.model,
                              to_file=str(self.diagram_path),
                              expand_nested=True, show_shapes=True, show_layer_names=False)

  # Trains model
  def train(self, epochs: int):
    # Creates train/test datasets using tf.data.Dataset
    train_ds, test_ds = create_dataset(self.dataset_folder, self.categories)
    # Callback (used in tf.keras.Model.fit) to save the model with the best validation accuracy
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      filepath=str(self.model_path),
      save_best_only=True,
      save_weights_only=False,
      monitor='val_loss',
      save_freq='epoch'
    )
    # Trains model using tf.keras.Model fit function
    self.model.fit(train_ds,
                   epochs=epochs,
                   validation_data=test_ds,
                   callbacks=[checkpoint],
                   verbose=2)


if __name__ == '__main__':
  # Parsing arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('folder', help='Dataset folder with tfrecords', type=str)
  parser.add_argument('--epochs', help='Epochs to train (Default: 100)', type=int)
  parser.add_argument('--use_regularizer', help='Use regularization', action="store_true")
  parser.add_argument('--use_batch_norm', help='Use batch normalization', action="store_true")
  args = parser.parse_args()
  epochs = args.epochs if args.epochs else 100
  use_regularizer = True if args.use_regularizer else False
  use_batch_norm = True if args.use_batch_norm else False
  # Dataset folder
  folder = args.folder
  # Creates model and trains
  learner = AirBinaryRNN(pathlib.Path(folder), use_regularizer=use_regularizer, use_batch_norm=use_batch_norm)
  learner.summary()
  learner.train(epochs)
  # Loads and evaluates model
  saved_model = tf.keras.models.load_model(str(learner.model_path))
  training_ds, testing_ds = create_dataset(learner.dataset_folder, learner.categories)
  display_performance(saved_model, training_ds, testing_ds)
