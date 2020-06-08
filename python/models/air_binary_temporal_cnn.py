#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import tensorflow as tf
from tfrecord_dataset import feature_description
import sys

# Constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
TIME_SIZE = 401
MFCC_SIZE = 128
BATCH_SIZE = 128
BUFFER_SIZE = 1000


# Parses observation from proto format and converts into correct format for training (input,output) = (spec,label)
@tf.function
def parse_observation(example: tf.Tensor) -> Tuple:
  observation = tf.io.parse_single_example(example, feature_description)
  mfcc = observation['mfcc']
  samples = observation['samples']
  spec = tf.transpose(tf.reshape(observation['spec'], (mfcc, samples)))
  label = observation['label'] == b'Airbus'

  return spec, label


# Creates dataset from tfrecord files
def create_dataset(train_record: str, test_record: str) -> Tuple:
  # Creates training data pipeline
  train_ds = tf.data.TFRecordDataset([train_record]).cache()
  train_ds = train_ds.map(parse_observation, num_parallel_calls=AUTOTUNE).cache()
  train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)

  # Creates test data pipeline
  test_ds = tf.data.TFRecordDataset([test_record])
  test_ds = test_ds.map(parse_observation, num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)

  return train_ds, test_ds


class AirBinaryTemporalCNN:
  """
  Simple CNN model with two 1D conv layers followed by global max pooling
  and a fully connected layer.

  Designed to perform binary classification on a simple dataset containing
  Airbus/Boeing aircraft take-off signals.
  """

  def __init__(self, regularize=True):
    # Stores options
    self.regularize = regularize
    # Builds model architecture
    self.model = self.build_model()
    # Selects loss and metric
    self.model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy']
    )

  # Builds models architecture returning a tf.keras.Model
  def build_model(self) -> tf.keras.Model:
    # A 1D convolutional layer looking for timeline pattern in the original
    # spectrogram. Filters have 50% overlap in the time axis.
    conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=16,
                                   strides=8, padding='same',
                                   activation=tf.nn.relu)
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.2)

    # Another 1D convolutional layer afterward to generate more complex
    # time features. The kernel combines analyzes three consecutive
    # activation maps from the previous layer output.
    conv2 = tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu)

    # Dropout layer
    dropout2 = tf.keras.layers.Dropout(0.2)

    # Performs global max pooling to "keep only the maximum" activation of
    # the previous convolutional layer filters. Technically answer where
    # (in time) the filter generated the strongest output.
    pooling1 = tf.keras.layers.GlobalMaxPooling1D()

    # Dense connecting layers to perform classification
    if self.regularize:
      dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                     kernel_regularizer=
                                     tf.keras.regularizers.l2(0.2))
    else:
      dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    # Create connections and returns model
    inputs = tf.keras.Input((TIME_SIZE, MFCC_SIZE))
    x = conv1(inputs)
    if self.regularize:
      x = dropout1(x)
    x = conv2(x)
    if self.regularize:
      x = dropout2(x)
    x = pooling1(x)
    outputs = dense1(x)

    return tf.keras.Model(inputs, outputs)

  # Prints out the model's summary
  def summary(self):
    self.model.summary()

  # Trains model
  def fit(self, train_record: str, test_record: str, epochs):
    # Creates train/test datasets using tf.data.Dataset
    train_ds, test_ds = create_dataset(train_record, test_record)

    # Trains model using tf.keras.Model fit function
    self.model.fit(train_ds, epochs=epochs, validation_data=test_ds, verbose=2)

  def save(self, path):
    self.model.save(path)


if __name__ == '__main__':
  # Parse arguments
  if len(sys.argv) < 3:
    train_file = '../exports/2020-02-03 11-55-25/train.tfrecord'
    test_file = '../exports/2020-02-03 11-55-25/test.tfrecord'
  else:
    train_file = sys.argv[1]
    test_file = sys.argv[2]

  # Creates model and trains
  model = AirBinaryTemporalCNN()
  model.summary()
  model.fit(train_file, test_file, 500)

  # Saves model
  model.save('trained_model/air_binary_temporal_cnn/')
