#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import datetime
import os
from typing import Tuple
import tensorflow as tf
from tfrecord_dataset import feature_description
from spectrogram_sequencer import SpectrogamSequencer
from utils import display_performance
from commons import AUTOTUNE, verify_tfrecords_from_directory, get_classes_from_directory

# Constants
TIME_SIZE = 401
MFCC_SIZE = 128
BATCH_SIZE = 64
BUFFER_SIZE = 1000
WINDOW_SIZE = 50
WINDOW_OVERLAP = 0.5


# Parses observation from proto format and converts into correct format for training (input,output) = (spec,label)
@tf.function
def parse_observation(example: tf.Tensor, categories: list) -> Tuple:
  observation = tf.io.parse_single_example(example, feature_description)
  mfcc = observation['mfcc']
  samples = observation['samples']
  spec = tf.reshape(observation['spec'], (mfcc, samples))
  spec = tf.expand_dims(spec, axis=-1)  # channel
  label = tf.argmax(tf.cast(
    tf.equal(observation['label'], tf.constant(categories)), dtype=tf.uint8
  ))
  return spec, label


# Creates dataset from tfrecord files
def create_dataset(dataset_folder: str, categories: list) -> Tuple:
  # Creates training data pipeline
  train_ds = tf.data.TFRecordDataset([dataset_folder + 'train.tfrecord'])
  train_ds = train_ds.map(lambda example: parse_observation(example, categories), num_parallel_calls=AUTOTUNE).cache()
  train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
  # Creates test data pipeline
  test_ds = tf.data.TFRecordDataset([dataset_folder + 'test.tfrecord'])
  test_ds = test_ds.map(lambda example: parse_observation(example, categories), num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)
  return train_ds, test_ds


class AirMulticlassRNN:
  """
  Simple recurrent neural network model using a sequence of spectrogram windows. The input of this model
  will be a sequence of overlapping spectrogram slices (windows).

  Designed to perform multinomial classification on a simple dataset
  containing four classes of aircraft take-off signals.
  """

  def __init__(self, dataset_folder: str, use_regularizer=True, use_batch_norm=False):
    # Verify train.tfrecord and test.tfrecord exist in dataset folder
    assert verify_tfrecords_from_directory(dataset_folder), 'There is no train.tfrecord or test.tfrecord' \
                                                            'in the folder specified '
    self.dataset_folder = dataset_folder
    self.experiments_folder = dataset_folder + 'experiments/' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S/')
    # Creates experiments folder if does not exist
    if not os.path.exists(self.experiments_folder):
      os.makedirs(self.experiments_folder)
    # Determine classes from folder
    categories = get_classes_from_directory(self.dataset_folder)
    # Verify binary
    assert len(categories) > 2, 'Wrong number of classes. Expecting more than two.'
    self.categories = categories
    # Stores options
    self.use_regularizer = use_regularizer
    self.use_batch_norm = use_batch_norm
    # Builds model architecture
    self.model = self.build_model()
    # Selects loss and metric
    self.model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy']
    )

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
    # Activation for first convolutional layer
    activation1 = tf.keras.layers.Activation(tf.nn.relu)
    # Pooling
    pooling1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3),
      name='Pool1'
    )
    # Second convolutional layer. Timed distributed to get applied to every segment of the inputted spectrogram.
    conv2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5,
                             padding='valid',
                             data_format='channels_last',
                             use_bias=False if self.use_batch_norm else True),
      name='Conv2'
    )
    # Batch norm layer (only used if requested)
    batch_norm2 = tf.keras.layers.BatchNormalization(axis=1, name='BatchNorm2')
    # Activation for first convolutional layer
    activation2 = tf.keras.layers.Activation(tf.nn.relu)
    # Third convolutional layer. Timed distributed.
    conv3 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5,
                             padding='valid',
                             data_format='channels_last',
                             use_bias=False if self.use_batch_norm else True),
      name='Conv3'
    )
    # Batch norm layer (only used if requested)
    batch_norm3 = tf.keras.layers.BatchNormalization(axis=1, name='BatchNorm3')
    # Activation for first convolutional layer
    activation3 = tf.keras.layers.Activation(tf.nn.relu)
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
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.3)
    # Recurrent layer to capture temporal relationships
    lstm1 = tf.keras.layers.LSTM(32, return_state=False, return_sequences=False)
    # Dropout layer
    dropout2 = tf.keras.layers.Dropout(0.2)
    # Dense to make the final classification
    dense1 = tf.keras.layers.Dense(len(self.categories), activation=tf.nn.softmax,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01) if self.use_regularizer else None,
                                   name='Dense')
    # Creates connections between layers of the model
    x = sequencer(inputs)
    x = conv1(x)
    if self.use_batch_norm:
      x = batch_norm1(x)
    x = pooling1(x)
    x = conv2(x)
    if self.use_batch_norm:
      x = batch_norm2(x)
    x = conv3(x)
    if self.use_batch_norm:
      x = batch_norm3(x)
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
    save_path = self.experiments_folder + 'diagrams/'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    tf.keras.utils.plot_model(self.model,
                              to_file=save_path + 'air_four_classes_rnn.jpg',
                              expand_nested=True, show_shapes=True)

  # Trains model
  def train(self, epochs: int):
    # Creates train/test datasets using tf.data.Dataset
    train_ds, test_ds = create_dataset(self.dataset_folder, self.categories)
    # Callback (used in tf.keras.Model.fit) to save the model with the best validation accuracy
    save_path = self.experiments_folder + 'trained_model/air_four_classes_rnn/'
    # Callback (used in tf.keras.Model.fit) to save the model with the best validation accuracy
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      filepath=save_path,
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

  def save(self, path):
    self.model.save(path)


if __name__ == '__main__':
  # Dataset folder
  folder = '../exports/2020-02-07 01-09-35 (four classes)/'
  # Creates model and trains
  learner = AirMulticlassRNN(folder, use_batch_norm=False)
  learner.summary()
  learner.train(100)
  # Loads and evaluates model
  saved_model = tf.keras.models.load_model(learner.experiments_folder + 'trained_model/air_four_classes_rnn/')
  training_ds, testing_ds = create_dataset(learner.dataset_folder, learner.categories)
  display_performance(saved_model, training_ds, testing_ds)
