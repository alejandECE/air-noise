#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import os
import argparse
import pathlib
import time
import air_siamese_logs as logging
from typing import Tuple
from spectrogram_sequencer import SpectrogamSequencer
import tensorflow as tf
import numpy as np
import commons
import sys
sys.path.append('../extraction')
from tfrecord_dataset import feature_description

# Constants
SIBLING_MODEL_NAME = 'air_siamense_sibling'
TIME_SIZE = 401
MFCC_SIZE = 128
POSITIVE_SAMPLES = 16
NEGATIVE_SAMPLES = 16
BUFFER_SIZE = 1000
WINDOW_SIZE = 50
WINDOW_OVERLAP = 0.5


# Contrastive loss
class ContrastiveLoss(tf.keras.losses.Loss):
  def __init__(self, margin: float = 1.):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def call(self, y_true, y_pred):
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    return y_true * tf.square(y_pred) + (1. - y_true) * tf.square(tf.maximum(0., self.margin - y_pred))


# Computes the Eucledian between two embeddings
class EucledianDistance(tf.keras.layers.Layer):
  def __init__(self):
    super(EucledianDistance, self).__init__()

  def call(self, emb1, emb2):
    return tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2), axis=-1, keepdims=True))


# Custom metric to evaluate current embeddings based on their distance and margin of contrastive loss
class EmbeddingQuality(tf.keras.metrics.Metric):
  def __init__(self, margin: float = 1., name='embedding_quality', **kwargs):
    super(EmbeddingQuality, self).__init__(name=name, **kwargs)
    self.margin = margin
    self.correct = self.add_weight(name='correct', initializer='zeros')
    self.observations = self.add_weight(name='observations', initializer='zeros')

  def update_state(self, y_true, y_pred, **kwargs):
    # Correct negative entries (distance >= margin)
    self.correct.assign_add(tf.reduce_sum(
      (1. - y_true) * tf.cast(tf.greater_equal(y_pred, self.margin), dtype=y_true.dtype)
    ))
    # Correct positive entries (distance <= margin)
    self.correct.assign_add(tf.reduce_sum(
      y_true * tf.cast(tf.greater_equal(self.margin, y_pred), dtype=y_true.dtype)
    ))
    # Total observations
    self.observations.assign_add(y_true.shape[0])

  def result(self):
    return self.correct / self.observations


# Computes the absolute difference between two embeddings
class AbsoluteDifference(tf.keras.layers.Layer):
  def __init__(self):
    super(AbsoluteDifference, self).__init__()

  def call(self, emb1, emb2):
    return tf.math.abs(emb1 - emb2)


# Parses observation from proto format
@tf.function
def parse_observation(example: tf.Tensor) -> Tuple:
  observation = tf.io.parse_single_example(example, feature_description)
  mfcc = observation['mfcc']
  samples = observation['samples']
  spec = tf.reshape(observation['spec'], (mfcc, samples))
  spec = tf.expand_dims(spec, axis=-1)  # channel
  label = observation['label']
  return spec, label


# Gets embedding by applying trained model
@tf.function
def get_embedding(sibling: tf.keras.Model, batch_inputs: tf.Tensor, batch_labels: tf.Tensor) -> Tuple:
  # Evaluates model to get embedding
  return sibling(batch_inputs), batch_labels


# Creates dataset of pairs for training only using observations from the categories listed
def create_training_dataset(tfrecords: list, categories: list) -> tf.data.Dataset:
  # Reads tfrecords from file
  records_ds = tf.data.TFRecordDataset(tfrecords)
  # Parses observation from tfrecords
  records_ds = records_ds.map(parse_observation, num_parallel_calls=commons.AUTOTUNE)
  # Creates list of datasets, each having observations from only one class
  labeled_datasets = []
  for category in categories:
    filtered = records_ds.filter(lambda spec, label: tf.equal(label, category))
    filtered = filtered.map(lambda spec, label: spec, num_parallel_calls=commons.AUTOTUNE).cache()
    labeled_datasets.append(filtered)

  # Creates positive observations
  # Creates list of datasets containing positive pairs (filtering out repeated pairs: (i,i) and (i,j) = (j,i))
  positive_datasets = []
  for dataset in labeled_datasets:
    shuffled = dataset.enumerate().shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    zipped = tf.data.Dataset.zip((shuffled, shuffled)).filter(lambda x, y: x[0] < y[0])
    zipped = zipped.map(lambda x, y: (x[1], y[1], tf.constant(1, dtype=tf.uint8)), num_parallel_calls=commons.AUTOTUNE)
    zipped = zipped.repeat()
    positive_datasets.append(zipped)
  # Samples the same number of pairs from all datasets
  samples_per_dataset = [int(POSITIVE_SAMPLES / len(positive_datasets))] * len(positive_datasets)
  for i in range(POSITIVE_SAMPLES % len(positive_datasets)):
    samples_per_dataset[i] += 1
  positive = positive_datasets[0].take(samples_per_dataset[0])
  for i in range(1, len(positive_datasets)):
    positive = positive.concatenate(positive_datasets[i].take(samples_per_dataset[i]))

  # Creates negative observations
  negative_datasets = []
  for i in range(len(labeled_datasets)):
    left_ds = labeled_datasets[i].shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).repeat()
    for j in range(i + 1, len(labeled_datasets)):
      right_ds = labeled_datasets[j].shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).repeat()
      zipped = tf.data.Dataset.zip((left_ds, right_ds)).map(lambda x, y: (x, y, tf.constant(0, dtype=tf.uint8)),
                                                            num_parallel_calls=commons.AUTOTUNE)
      negative_datasets.append(zipped)
  # Samples the same number of pairs from all datasets
  samples_per_dataset = [int(NEGATIVE_SAMPLES / len(negative_datasets))] * len(negative_datasets)
  for i in range(NEGATIVE_SAMPLES % len(negative_datasets)):
    samples_per_dataset[i] += 1
  negative = negative_datasets[0].take(samples_per_dataset[0])
  for i in range(1, len(negative_datasets)):
    negative = negative.concatenate(negative_datasets[i].take(samples_per_dataset[i]))

  # Combines positive and negative examples
  train_ds = positive.concatenate(negative)
  train_ds = train_ds.batch(POSITIVE_SAMPLES + NEGATIVE_SAMPLES).repeat().prefetch(1)

  return train_ds


# Compute the squared distance between two embeddings
def compute_squared_distance(target_embedding: tf.Tensor, embeddings_batch: tf.Tensor):
  return ((target_embedding.numpy() - embeddings_batch.numpy()) ** 2).sum(axis=1)


# Creates dataset for n-way one shot learning for testing using the categories listed only
def create_test_dataset(tfrecords: list, categories: list, sibling: tf.keras.Model, n_way: int):
  # Reads tfrecords from file
  records_ds = tf.data.TFRecordDataset(tfrecords)
  # Parses observation from tfrecords
  records_ds = records_ds.map(parse_observation, num_parallel_calls=commons.AUTOTUNE)
  # Define batches
  records_ds = records_ds.batch(POSITIVE_SAMPLES + NEGATIVE_SAMPLES)
  # Get the embedding dataset using the model trained
  embedding_ds = records_ds.map(lambda inputs, labels: get_embedding(sibling, inputs, labels),
                                num_parallel_calls=commons.AUTOTUNE).unbatch().cache()
  # Creates a dictionary of datasets with (keys, values): label, (positive samples, negative samples)
  datasets = {}
  for category in categories:
    # Creates a dataset only selecting observations from the category
    positive_ds = embedding_ds.filter(lambda embedding, label: tf.equal(label, category))
    positive_ds = positive_ds.map(lambda embedding, label: embedding, num_parallel_calls=commons.AUTOTUNE)
    positive_ds = positive_ds.shuffle(BUFFER_SIZE).prefetch(1)
    # Creates dataset with the rest of observations
    negative_ds = embedding_ds.filter(lambda embedding, label: tf.not_equal(label, category))
    negative_ds = negative_ds.shuffle(BUFFER_SIZE).batch(n_way - 1, drop_remainder=True).prefetch(1)
    datasets[category] = (positive_ds, negative_ds)
  return datasets


class AirSiameseLearner:
  """
  Siamese architecture to learn embeddings that allow to differentiate aircraft classes based on noise during take-off.
  """

  def __init__(self, dataset_folder: pathlib.Path, leftout:int = 0,
               use_regularizer=True,
               use_batch_norm=False,
               architecture='absolute'):
    # Setups experiment folder
    self.dataset_folder = dataset_folder
    self.experiment_path, self.model_path, self.diagram_path = self.setup_experiment_folder()
    # Determine classes from folder
    self.categories = commons.get_classes_from_folder(dataset_folder)
    self.leftout = leftout
    # Stores options
    self.architecture = architecture
    self.use_regularization = use_regularizer
    self.use_batch_norm = use_batch_norm
    # Builds model architecture
    self.loss = None
    self.optimizer = None
    self.metric = None
    self.siamese, self.sibling = self.build_siamese_model()

  # Setup experiment folder
  def setup_experiment_folder(self) -> Tuple:
    # Experiment path (root folder of the experiment)
    experiment_path = commons.generate_experiment_path(self.dataset_folder)
    # Model path (where it is saved during training)
    model_path = commons.generate_model_path(experiment_path, SIBLING_MODEL_NAME)
    # Keras model diagram
    diagram_path = commons.generate_diagram_path(experiment_path, SIBLING_MODEL_NAME)
    # Returns all relevant paths as tuple
    return experiment_path, model_path, diagram_path

  # Builds siamese network and sibling network
  def build_siamese_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
    # Inputs to the siamese model. Data is fed as a pair (input1, input2)
    input1 = tf.keras.layers.Input((MFCC_SIZE, TIME_SIZE, 1), name='spec1')
    input2 = tf.keras.layers.Input((MFCC_SIZE, TIME_SIZE, 1), name='spec2')
    # Actual model receiving a spectrogram and outputting an embedding
    sibling = self.build_sibling_model()
    # Embeddings for both inputs
    embedding1 = sibling(input1)
    embedding2 = sibling(input2)
    if self.architecture == 'absolute':
      # Absolute difference and dense layer to output probability of been the same or different class
      similarity = AbsoluteDifference()(embedding1, embedding2)
      output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(similarity)
    else:
      # Computes Eucledian distance between embeddings
      output = EucledianDistance()(embedding1, embedding2)
    # Builds siamese architecture
    siamese = tf.keras.models.Model([input1, input2], output, name='siamese_nn')
    # Selects loss and optimizer
    self.optimizer = tf.keras.optimizers.Adam()
    if self.architecture == 'absolute':
      self.metric = tf.keras.metrics.BinaryAccuracy()
      self.loss = tf.keras.losses.BinaryCrossentropy()
    else:
      self.metric = EmbeddingQuality(margin=1.)
      self.loss = ContrastiveLoss(margin=1.)
    # Return tuple of models
    return siamese, sibling

  # Builds sibling model architecture returning a tf.keras.Model
  def build_sibling_model(self) -> tf.keras.Model:
    # Create inputs
    spectrogram = tf.keras.layers.Input((MFCC_SIZE, TIME_SIZE, 1))
    # Creates spectrogram sequencer layer
    sequencer = SpectrogamSequencer(WINDOW_SIZE, WINDOW_OVERLAP)
    # First convolutional layer to find spatial features in a segment of the spectrogram. Timed distributed to get
    # applied to every segment of the inputted spectrogram.
    conv1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5, padding='valid',
                             data_format='channels_last',
                             use_bias=False if self.use_batch_norm else True,
                             name='Conv1')
    )
    # Batch norm layer (only used if requested)
    batch_norm1 = tf.keras.layers.BatchNormalization(axis=1, name='BatchNorm1')
    # Activation for Conv1
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
    # Activation for second convolutional layer
    activation2 = tf.keras.layers.Activation(tf.nn.relu, name='Relu2')
    # Third convolutional layer. Timed distributed.
    conv3 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5,
                             padding='valid',
                             data_format='channels_last',
                             use_bias=False if self.use_batch_norm else True,
                             name='Conv3')
    )
    # Batch norm layer (only used if requested)
    batch_norm3 = tf.keras.layers.BatchNormalization(axis=1, name='BatchNorm3')
    # Activation for third convolutional layer
    activation3 = tf.keras.layers.Activation(tf.nn.relu, name='Relu3')
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
    lstm1 = tf.keras.layers.LSTM(32, return_state=False, return_sequences=False)
    # Creates connections between layers of the model
    x = sequencer(spectrogram)
    x = conv1(x)
    if self.use_batch_norm:
      x = batch_norm1(x)
    x = activation1(x)
    x = pooling1(x)
    x = conv2(x)
    if self.use_batch_norm:
      x = batch_norm2(x)
    x = activation2(x)
    x = conv3(x)
    if self.use_batch_norm:
      x = batch_norm3(x)
    x = activation3(x)
    x = pooling2(x)
    x = flatten1(x)
    if self.use_regularization:
      x = dropout1(x)
    embedding = lstm1(x)
    return tf.keras.Model(spectrogram, embedding, name='sibling_nn')

  # Prints out the model's summary
  def summary(self) -> None:
    self.siamese.summary()
    self.sibling.summary()
    tf.keras.utils.plot_model(self.sibling,
                              to_file=str(self.diagram_path),
                              expand_nested=True, show_shapes=True, show_layer_names=False)

  @tf.function
  def train_step(self, batch):
    # Unwrapped inputs and expected output
    input1, input2, y_true = batch
    with tf.GradientTape() as tape:
      # Computes siamese architecture output
      y_pred = self.siamese([input1, input2])
      # Computes loss in batch
      loss = self.loss(y_true, tf.squeeze(y_pred))
    # Determines gradients
    variables = self.siamese.trainable_variables
    gradients = tape.gradient(loss, variables)
    # Updates params
    self.optimizer.apply_gradients(zip(gradients, variables))
    # Computes metric
    metric = self.metric(tf.cast(y_true, dtype=y_pred.dtype), tf.squeeze(y_pred))
    self.metric.reset_states()
    # Returns loss and metric
    return loss, metric

  # Trains model
  def train(self, epochs: int, verbose=False) -> None:
    # Creates dataset leaving out the least common classes
    included_categories = self.categories[:-self.leftout] if self.leftout > 0 else self.categories
    train_ds = create_training_dataset(
      commons.get_tfrecords_from_folder(self.dataset_folder),
      included_categories
    )
    # Creates an iterator to request batches
    iterator = iter(train_ds)
    # Training loop
    log_entries = []
    for epoch in range(epochs):
      start = time.perf_counter()
      # Gets a batch from dataset
      batch = iterator.get_next()
      # Performs update step
      loss, metric = self.train_step(batch)
      # Logs training step
      entry = logging.TrainingEntry(
        current=epoch + 1,
        epochs=epochs,
        time=time.perf_counter() - start,
        loss=loss,
        metric=tf.squeeze(metric)
      )
      log_entries.append(entry)
      if verbose:
        logging.write_step_log_to_console(entry)
    # Generates summary
    excluded_categories = self.categories[-self.leftout:] if self.leftout > 0 else []
    summary = logging.TrainingSummary(
      included=included_categories,
      excluded=excluded_categories
    )
    if verbose:
      logging.write_summary_log_to_console(summary)
    # Logs training to file
    with open(self.experiment_path / 'training.txt', 'w') as log_file:
      # Logging each step
      logging.write_step_log_to_file(log_file, log_entries)
      # Logging summary
      logging.write_summary_log_to_file(log_file, summary)
    # Stores resulting sibling network
    tf.keras.models.save_model(self.sibling, str(self.model_path))

  # Use n-way one shot learning evaluation
  def evaluate(self, evals: int = 100, nway=4, verbose=False) -> None:
    # Creates dataset to sample from
    filtered_categories = self.categories[-self.leftout:] if self.leftout > 0 else self.categories
    datasets = create_test_dataset(
      commons.get_tfrecords_from_folder(self.dataset_folder),
      filtered_categories,
      self.sibling,
      nway
    )
    # Performs n-way one shot learning
    log_entries = []
    # Randomly samples from excluded categories (comparisons are still performed vs all categories)
    sampled_categories = np.random.choice(filtered_categories, evals, replace=True)
    for one_shot_category in sampled_categories:
      start = time.perf_counter()
      # Gets datasets for evaluating current category
      positive_ds, negative_ds = datasets[one_shot_category]
      iter_positive = iter(positive_ds)
      iter_negative = iter(negative_ds)
      # Gets the one shot observation to learn from
      one_shot_obs = iter_positive.get_next()
      # Compares the one shot embedding with one sample drawn randomly from the same class
      positive_obs = iter_positive.get_next()
      comparisons = [
        logging.Comparison(
          distance=compute_squared_distance(one_shot_obs, tf.expand_dims(positive_obs, axis=0))[0],
          category=one_shot_category
        )
      ]
      # Compares the one shot embedding with n - 1 samples from drawn randomly from the other classes
      negative_obs, negative_labels = iter_negative.get_next()
      negative_distances = compute_squared_distance(one_shot_obs, negative_obs)
      for value, label in zip(negative_distances, negative_labels):
        comparisons.append(
          logging.Comparison(
            distance=value,
            category=label.numpy()
          )
        )
      end = time.perf_counter()
      # Generates evaluation entry
      entry = logging.EvaluationEntry(
        time=end - start,
        category=one_shot_category,
        comparisons=sorted(comparisons, key=lambda comparison: comparison.distance)
      )
      log_entries.append(entry)
      if verbose:
        logging.write_step_log_to_console(entry)
    # Generates summary
    accuracies = []
    for i in range(1, nway // 2 + 1):
      matches = [
        any([comparison.category == entry.category for comparison in entry.comparisons[:i]])
        for entry in log_entries
      ]
      accuracies.append(sum(matches) / len(matches))
    summary = logging.EvaluationSummary(accuracies=accuracies)
    logging.write_summary_log_to_console(summary)
    # Logs evaluation to file
    with open(self.experiment_path / 'evaluation.txt', 'w') as log_file:
      # Logging each step
      logging.write_step_log_to_file(log_file, log_entries)
      # Logging summary
      logging.write_summary_log_to_file(log_file, summary)


if __name__ == '__main__':
  # Defining arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('folder', help='Dataset folder with tfrecords', type=str)
  parser.add_argument('--epochs', help='Epochs to train (Default: 100)', type=int)
  parser.add_argument('--leftout', help='Categories leftout (Default: 0)', type=int)
  parser.add_argument('--contrastive', help='Use contrastive loss', action="store_true")
  parser.add_argument('--evals', help='Number of evaluations once trained (Default: 100)', type=int)
  parser.add_argument('--nway', help='N-way used for evaluation (Default: 8)', type=int)
  parser.add_argument('--use_regularizer', help='Use regularization', action="store_true")
  parser.add_argument('--use_batch_norm', help='Use batch normalization', action="store_true")
  parser.add_argument('--verbose', help='Verbosity', action="store_true")
  parser.add_argument('--cpu', help='Use CPU instead of GPU', action="store_true")
  # Parsing arguments
  args = parser.parse_args()
  epochs = args.epochs if args.epochs else 100
  leftout = args.leftout if args.leftout else 0
  evals = args.evals if args.evals else 100
  nway = args.nway if args.nway else 8
  use_regularizer = True if args.use_regularizer else False
  use_batch_norm = True if args.use_batch_norm else False
  verbose = True if args.verbose else False
  contrastive = True if args.verbose else False
  if args.cpu:
    # Selecting CPU by not exposing GPU devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  else:
    # Allowing memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
  # Dataset folder
  folder = args.folder
  # Performs training using only the first four classes
  learner = AirSiameseLearner(pathlib.Path(folder),
                              leftout=leftout,
                              architecture='contrastive' if contrastive else 'absolute',
                              use_regularizer=use_regularizer,
                              use_batch_norm=use_batch_norm)
  learner.summary()
  learner.train(epochs, verbose=verbose)
  # Evaluates the model using n-way one shot learning
  learner.evaluate(evals=evals, nway=nway, verbose=verbose)
