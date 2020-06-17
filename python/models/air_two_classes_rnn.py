#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import tensorflow as tf
from tfrecord_dataset import feature_description
from spec_sequencer import SpecSequencer
from utils import display_performance

# Constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
TIME_SIZE = 401
MFCC_SIZE = 128
WINDOW_SIZE = 50
WINDOW_OVERLAP = 0.5
BATCH_SIZE = 64
BUFFER_SIZE = 1000


# Creates a sequence of windows from a spectrogram
@tf.function
def build_sequence(spec, label):
  samples = tf.shape(spec)[1]
  start = tf.range(0, samples - WINDOW_SIZE, int(WINDOW_SIZE * WINDOW_OVERLAP))
  end = tf.range(WINDOW_SIZE, samples, int(WINDOW_SIZE * WINDOW_OVERLAP))
  sequence = tf.map_fn(lambda index: spec[:, index[0]:index[1]],
                       tf.stack([start, end], axis=1),
                       back_prop=False,
                       dtype=tf.float32)

  return sequence, label


# Parses observation from proto format and converts into correct format for training (input,output) = (spec,label)
@tf.function
def parse_observation(example: tf.Tensor) -> Tuple:
  observation = tf.io.parse_single_example(example, feature_description)
  mfcc = observation['mfcc']
  samples = observation['samples']
  spec = tf.reshape(observation['spec'], (mfcc, samples))
  spec = tf.expand_dims(spec, axis=-1)  # channel
  label = observation['label'] == b'Airbus'

  return spec, label


# Creates dataset from tfrecord files
def create_dataset(train_record: str, test_record: str, sequencer=False) -> Tuple:
  # Creates training data pipeline
  train_ds = tf.data.TFRecordDataset([train_record])
  train_ds = train_ds.map(parse_observation, num_parallel_calls=AUTOTUNE).cache()
  if not sequencer:
    train_ds = train_ds.map(build_sequence, num_parallel_calls=AUTOTUNE).cache()
  train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)

  # Creates test data pipeline
  test_ds = tf.data.TFRecordDataset([test_record])
  test_ds = test_ds.map(parse_observation, num_parallel_calls=AUTOTUNE).cache()
  if not sequencer:
    test_ds = test_ds.map(build_sequence, num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)

  return train_ds, test_ds


class AirBinaryRNN:
  """
  Simple recurrent neural network model using a sequence of spectrogram windows. The input of this model
  will be a sequence of overlapping spectrogram slices (windows).

  Designed to perform binary classification on a simple dataset containing
  Airbus/Boeing aircraft take-off signals.
  """

  def __init__(self, regularize=True, batch_norm=False, sequencer=False):
    # Stores options
    self.regularize = regularize
    self.sequencer = sequencer
    self.batch_norm = batch_norm
    # Builds model architecture
    self.model = self.build_model()
    # Selects loss and metric
    self.model.compile(
      optimizer=tf.keras.optimizers.Adam(decay=1e-4),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy']
    )

  # Builds models architecture returning a tf.keras.Model
  def build_model(self) -> tf.keras.Model:
    # Create inputs
    if self.sequencer:
      inputs = tf.keras.layers.Input((MFCC_SIZE, TIME_SIZE, 1))
    else:
      inputs = tf.keras.layers.Input((None, MFCC_SIZE, WINDOW_SIZE, 1))
    # Creates the sequencer layer (only used if requested)
    sequencer1 = SpecSequencer(WINDOW_SIZE, WINDOW_OVERLAP)
    # First convolutional layer to find spatial features in a segment of the spectrogram. Timed distributed to get
    # applied to every segment of the inputted spectrogram.
    conv1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5, padding='valid', activation=tf.nn.relu, data_format='channels_last'),
      name='Conv1'
    )

    # Pooling
    pooling1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3),
      name='Pool1'
    )
    # Batch norm layer (only used if requested)
    batch_norm1 = tf.keras.layers.BatchNormalization(axis=1, name='BatchNorm1')
    # Second convolutional layer. Timed distributed to get applied to every segment of the inputted spectrogram.
    conv2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5,
                             padding='valid',
                             activation=tf.nn.relu,
                             data_format='channels_last'),
      name='Conv2'
    )
    # Pooling
    pooling2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3),
      name='Pool2'
    )
    # Flatten results from convolutions/pooling to be fed to the lstm layers
    flatten1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Flatten(),
      name='Flatten'
    )
    # Batch norm layer (only used if requested)
    batch_norm2 = tf.keras.layers.BatchNormalization(axis=1, name='BatchNorm2')
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.3)
    # Recurrent layer to capture temporal relationships
    lstm1 = tf.keras.layers.LSTM(32,
                                 return_state=False,
                                 return_sequences=False)
    # Dropout layer
    dropout2 = tf.keras.layers.Dropout(0.1)
    # Dense to make the final classification
    dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.3) if self.regularize else None,
                                   name='Dense')
    # Creates connections between layers of the model
    if self.sequencer:
      x = sequencer1(inputs)
      x = conv1(x)
    else:
      x = conv1(inputs)
    x = pooling1(x)
    if self.batch_norm:
      x = batch_norm1(x)
    x = conv2(x)
    x = pooling2(x)
    x = flatten1(x)
    if self.batch_norm:
      x = batch_norm2(x)
    if self.regularize:
      x = dropout1(x)
    x = lstm1(x)
    if self.regularize:
      x = dropout2(x)
    outputs = dense1(x)
    return tf.keras.Model(inputs, outputs)

  # Prints out the model's summary
  def summary(self) -> None:
    self.model.summary()
    tf.keras.utils.plot_model(self.model,
                              to_file='diagrams/air_two_classes_rnn.jpg',
                              expand_nested=True, show_shapes=True)

  # Trains model
  def fit(self, train_record: str, test_record: str, epochs: int):
    # Creates train/test datasets using tf.data.Dataset
    train_ds, test_ds = create_dataset(train_record, test_record, self.sequencer)

    # Callback (used in tf.keras.Model.fit) to save the model with the best validation accuracy
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      filepath='trained_model/air_two_classes_rnn/',
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

  def save(self, path: str) -> None:
    self.model.save(path)


if __name__ == '__main__':
  # Dataset files
  train_file = '../exports/2020-02-03 11-55-25/train.tfrecord'
  test_file = '../exports/2020-02-03 11-55-25/test.tfrecord'

  # Creates model and trains
  model = AirBinaryRNN(sequencer=True, batch_norm=True)
  model.summary()
  model.fit(train_file, test_file, 100)

  # Loads and evaluates model
  saved = tf.keras.models.load_model('trained_model/air_two_classes_rnn/')
  training_ds, testing_ds = create_dataset(train_file, test_file, sequencer=True)
  display_performance(saved, training_ds, testing_ds)
