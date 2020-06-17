#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import tensorflow as tf
from tfrecord_dataset import feature_description
from utils import display_performance

# Constants
AIRCRAFT_LABELS = [b'A320-2xx (CFM56-5)', b'B737-7xx (CF56-7B22-)', b'ERJ190 (CF34-10E)', b'B737-8xx (CF56-7B22+)']
AUTOTUNE = tf.data.experimental.AUTOTUNE
TIME_SIZE = 401
MFCC_SIZE = 128
BATCH_SIZE = 64
BUFFER_SIZE = 1000


# Parses observation from proto format and converts into correct format for training (input,output) = (spec,label)
@tf.function
def parse_observation(example: tf.Tensor) -> Tuple:
  observation = tf.io.parse_single_example(example, feature_description)
  mfcc = observation['mfcc']
  samples = observation['samples']
  spec = tf.transpose(tf.reshape(observation['spec'], (mfcc, samples)))
  label = tf.argmax(tf.cast(
    tf.equal(observation['label'], tf.constant(AIRCRAFT_LABELS)), dtype=tf.uint8
  ))

  return spec, label


# Creates dataset from tfrecord files
def create_dataset(train_record: str, test_record: str) -> Tuple:
  # Creates training data pipeline
  train_ds = tf.data.TFRecordDataset([train_record])
  train_ds = train_ds.map(parse_observation, num_parallel_calls=AUTOTUNE).cache()
  train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)

  # Creates test data pipeline
  test_ds = tf.data.TFRecordDataset([test_record])
  test_ds = test_ds.map(parse_observation, num_parallel_calls=AUTOTUNE).cache()
  test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)

  return train_ds, test_ds


class AirMulticlassTemporalCNN:
  """
  Simple CNN model with two 1D conv layers followed by global max pooling
  and a fully connected layer.

  Designed to perform multiclass classification on a simple dataset
  containing four classes of aircraft take-off signals.
  """

  def __init__(self, categories: int, regularize=True, batch_norm=False):
    # Stores options
    self.categories = categories
    self.regularize = regularize
    self.batch_norm = batch_norm
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
    # Creates inputs
    inputs = tf.keras.Input((TIME_SIZE, MFCC_SIZE))
    # A 1D convolutional layer looking for timeline pattern in the original
    # spectrogram. Filters have 50% overlap in the time axis.
    conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=16,
                                   strides=8, padding='same',
                                   activation=tf.nn.relu)
    # Batch norm layer
    batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1, name='BatchNorm1')
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.3)
    # Another 1D convolutional layer afterward to generate more complex
    # time features. The kernel combines analyzes three consecutive
    # activation maps from the previous layer output.
    conv2 = tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu)
    # Batch norm layer
    batch_norm2 = tf.keras.layers.BatchNormalization(axis=-1, name='BatchNorm2')
    # Dropout layer
    dropout2 = tf.keras.layers.Dropout(0.1)
    # Performs global max pooling to "keep only the maximum" activation of
    # the previous convolutional layer filters. Technically answer where
    # (in time) the filter generated the strongest output.
    pooling1 = tf.keras.layers.GlobalMaxPooling1D()
    # Dense connecting layers to perform classification
    dense1 = tf.keras.layers.Dense(self.categories, activation=tf.nn.softmax,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)) if self.regularize else None
    # Creates connections between layers of the model
    x = conv1(inputs)
    if self.batch_norm:
      x = batch_norm1(x)
    if self.regularize:
      x = dropout1(x)
    x = conv2(x)
    if self.batch_norm:
      x = batch_norm2(x)
    if self.regularize:
      x = dropout2(x)
    x = pooling1(x)
    outputs = dense1(x)
    return tf.keras.Model(inputs, outputs)

  # Prints out the model's summary
  def summary(self) -> None:
    self.model.summary()
    tf.keras.utils.plot_model(self.model,
                              to_file='diagrams/air_four_classes_temporal_cnn.jpg',
                              expand_nested=True, show_shapes=True)

  # Trains model
  def fit(self, train_record: str, test_record: str, epochs: int):
    # Creates train/test datasets using tf.data.Dataset
    train_ds, test_ds = create_dataset(train_record, test_record)

    # Callback (used in tf.keras.Model.fit) to save the model with the best validation accuracy
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      filepath='trained_model/air_four_classes_temporal_cnn/',
      save_best_only=True,
      save_weights_only=False,
      monitor='val_accuracy',
      save_freq='epoch'
    )

    # Trains model using tf.keras.Model fit function
    self.model.fit(train_ds,
                   epochs=epochs,
                   validation_data=test_ds,
                   callbacks=[checkpoint],
                   verbose=2)

    # Show confusion matrix, accuracy, recall, precision and f1score
    display_performance(self.model, train_ds, test_ds)

  def save(self, path: str) -> None:
    self.model.save(path)


if __name__ == '__main__':
  # Dataset files
  train_file = '../exports/2020-02-07 01-09-35/train.tfrecord'
  test_file = '../exports/2020-02-07 01-09-35/test.tfrecord'

  # Creates model and trains
  model = AirMulticlassTemporalCNN(4, batch_norm=False)
  model.summary()
  model.fit(train_file, test_file, 200)

  # Loads and evaluates model
  saved = tf.keras.models.load_model('trained_model/air_four_classes_temporal_cnn/')
  training_ds, testing_ds = create_dataset(train_file, test_file)
  display_performance(saved, training_ds, testing_ds)
