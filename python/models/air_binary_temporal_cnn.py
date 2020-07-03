#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import datetime
import argparse
import pathlib
from typing import Tuple
import tensorflow as tf
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
MODELS_NAME = 'air_binary_temporal_cnn'
TIME_SIZE = 401
MFCC_SIZE = 128
BATCH_SIZE = 128
BUFFER_SIZE = 1000


# Parses observation from proto format and converts into correct format for training (input,output) = (spec,label)
@tf.function
def parse_observation(example: tf.Tensor, categories: list) -> Tuple:
  observation = tf.io.parse_single_example(example, feature_description)
  mfcc = observation['mfcc']
  samples = observation['samples']
  spec = tf.transpose(tf.reshape(observation['spec'], (mfcc, samples)))
  label = observation['label'] == categories[0]
  return spec, label


# Creates dataset from tfrecord files
def create_dataset(dataset_folder: pathlib.Path, categories: list) -> Tuple:
  # Creates training data pipeline
  train_ds = tf.data.TFRecordDataset([str(pathlib.Path(dataset_folder) / 'train.tfrecord')])
  train_ds = train_ds.map(lambda example: parse_observation(example, categories), num_parallel_calls=AUTOTUNE).cache()
  train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)
  # Creates test data pipeline
  test_ds = tf.data.TFRecordDataset([str(pathlib.Path(dataset_folder) / 'test.tfrecord')])
  test_ds = test_ds.map(lambda example: parse_observation(example, categories), num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)
  return train_ds, test_ds


class AirBinaryTemporalCNN:
  """
  Simple CNN model with two 1D conv layers followed by global max pooling
  and a fully connected layer.

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
    # Creates inputs
    inputs = tf.keras.Input((TIME_SIZE, MFCC_SIZE))
    # A 1D convolutional layer looking for timeline pattern in the original
    # spectrogram. Filters have 50% overlap in the time axis.
    conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=16,
                                   strides=8, padding='same',
                                   use_bias=False if self.use_batch_norm else True,
                                   name='Conv1')
    # Batch norm layer
    batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1, name='BatchNorm1')
    # Activation for the first convolutional layer
    activation1 = tf.keras.layers.Activation(tf.nn.relu)
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.3)
    # Another 1D convolutional layer afterward to generate more complex
    # time features. The kernel combines analyzes three consecutive
    # activation maps from the previous layer output.
    conv2 = tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                   padding='same',
                                   use_bias=False if self.use_batch_norm else True,
                                   name='Conv2')
    # Batch norm layer
    batch_norm2 = tf.keras.layers.BatchNormalization(axis=-1, name='BatchNorm2')
    # Activation for the first convolutional layer
    activation2 = tf.keras.layers.Activation(tf.nn.relu)
    # Dropout layer
    dropout2 = tf.keras.layers.Dropout(0.3)
    # Performs global max pooling to "keep only the maximum" activation of
    # the previous convolutional layer filters. Technically answer where
    # (in time) the filter generated the strongest output.
    pooling1 = tf.keras.layers.GlobalAveragePooling1D()
    # Dense connecting layers to perform classification
    dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.3)) if self.use_regularizer else None
    # Creates connections between layers of the model
    x = conv1(inputs)
    if self.use_batch_norm:
      x = batch_norm1(x)
    x = activation1(x)
    if self.use_regularizer:
      x = dropout1(x)
    x = conv2(x)
    if self.use_batch_norm:
      x = batch_norm2(x)
    x = activation2(x)
    if self.use_regularizer:
      x = dropout2(x)
    x = pooling1(x)
    outputs = dense1(x)
    return tf.keras.Model(inputs, outputs)

  # Prints out the model's summary
  def summary(self):
    self.model.summary()
    tf.keras.utils.plot_model(self.model,
                              to_file=str(self.diagram_path),
                              expand_nested=True, show_shapes=True, show_layer_names=False)

  # Trains model
  def train(self, epochs):
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
  learner = AirBinaryTemporalCNN(pathlib.Path(folder), use_regularizer=use_regularizer, use_batch_norm=use_batch_norm)
  learner.summary()
  learner.train(epochs)
  # Loads and evaluates model
  saved_model = tf.keras.models.load_model(str(learner.model_path))
  training_ds, testing_ds = create_dataset(learner.dataset_folder, learner.categories)
  display_performance(saved_model, training_ds, testing_ds)

