#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from commons import AIRCRAFT_EIGHT_LABELS
from commons import AUTOTUNE
from typing import Tuple
from tfrecord_dataset import feature_description
import tensorflow as tf
import numpy as np
import time
import datetime
import os

TIME_SIZE = 401
MFCC_SIZE = 128
POSITIVE_SAMPLES = 32
NEGATIVE_SAMPLES = 32
BUFFER_SIZE = 1000
WINDOW_SIZE = 50
WINDOW_OVERLAP = 0.5


# Computes the absolute difference of two embeddings
class AbsoluteSimilarity(tf.keras.layers.Layer):
  def __init__(self):
    super(AbsoluteSimilarity, self).__init__()

  def call(self, emb1, emb2):
    return tf.math.abs(emb1 - emb2)


# Computes the chi-squared similarity between two embeddings
class ChiSquaredSimilarity(tf.keras.layers.Layer):
  def __init__(self):
    super(AbsoluteSimilarity, self).__init__()

  def call(self, emb1, emb2):
    return tf.math.square(emb1 - emb2) / (emb1 + emb2)


# Creates a sequence of windows from a spectrogram
@tf.function
def build_sequence(spec: tf.Tensor, label: tf.Tensor) -> Tuple:
  samples = tf.shape(spec)[1]
  start = tf.range(0, samples - WINDOW_SIZE, int(WINDOW_SIZE * WINDOW_OVERLAP))
  end = tf.range(WINDOW_SIZE, samples, int(WINDOW_SIZE * WINDOW_OVERLAP))
  sequence = tf.map_fn(lambda index: spec[:, index[0]:index[1]],
                       tf.stack([start, end], axis=1),
                       back_prop=False,
                       dtype=tf.float32)
  return sequence, label


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
def get_embedding(model_path: str, batch_inputs: tf.Tensor, batch_labels: tf.Tensor, ) -> Tuple:
  # Load trained sibling model
  sibling = tf.keras.models.load_model(model_path)
  # Evaluates model to get embedding
  return sibling(batch_inputs), batch_labels


# Creates dataset of pairs for training only using observations from the categories listed
def create_training_dataset(records_path: str, categories: list) -> tf.data.Dataset:
  # Reads tfrecords from file
  records_ds = tf.data.TFRecordDataset([records_path])
  # Parses observation from tfrecords
  records_ds = records_ds.map(parse_observation, num_parallel_calls=AUTOTUNE)
  # Builds a sequence to feed siamese networks
  records_ds = records_ds.map(build_sequence, num_parallel_calls=AUTOTUNE)
  # Creates list of datasets, each having observations from only one class
  labeled_datasets = []
  for category in categories:
    filtered = records_ds.filter(lambda spec, label: tf.equal(label, category))
    filtered = filtered.map(lambda spec, label: spec, num_parallel_calls=AUTOTUNE).cache()
    labeled_datasets.append(filtered)

  # Creates positive observations
  # Creates list of datasets containing positive pairs (filtering out repeated pairs: (i,i) and (i,j) = (j,i))
  positive_datasets = []
  for dataset in labeled_datasets:
    shuffled = dataset.enumerate().shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    zipped = tf.data.Dataset.zip((shuffled, shuffled)).filter(lambda x, y: x[0] < y[0])
    zipped = zipped.map(lambda x, y: (x[1], y[1], tf.constant(1, dtype=tf.uint8)), num_parallel_calls=AUTOTUNE)
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
                                                            num_parallel_calls=AUTOTUNE)
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
  return ((target_embedding.numpy() - embeddings_batch.numpy())**2).sum(axis=1)


# Creates dataset for n-way one shot learning for testing using the categories listed only
def create_test_dataset(model_path: str, records_path: str, categories: list, n_way: int):
  # Reads tfrecords from file
  records_ds = tf.data.TFRecordDataset([records_path])
  # Parses observation from tfrecords
  records_ds = records_ds.map(parse_observation, num_parallel_calls=AUTOTUNE)
  # Builds a sequence to feed siamese networks
  records_ds = records_ds.map(build_sequence, num_parallel_calls=AUTOTUNE).batch(POSITIVE_SAMPLES + NEGATIVE_SAMPLES)
  # Get the embedding dataset using the model trained
  embedding_ds = records_ds.map(lambda inputs, labels: get_embedding(model_path, inputs, labels),
                                num_parallel_calls=AUTOTUNE).unbatch().cache()
  # Creates a dictionary of datasets with (keys, values): label, (positive samples, negative samples)
  datasets = {}
  for category in categories:
    # Creates a dataset only selecting observations from the category
    positive_ds = embedding_ds.filter(lambda embedding, label: tf.equal(label, category))
    positive_ds = positive_ds.map(lambda embedding, label: embedding, num_parallel_calls=AUTOTUNE)
    positive_ds = positive_ds.shuffle(BUFFER_SIZE).prefetch(4)
    # Creates dataset with the rest of observations
    negative_ds = embedding_ds.filter(lambda embedding, label: tf.not_equal(label, category))
    negative_ds = negative_ds.shuffle(BUFFER_SIZE).batch(n_way - 1, drop_remainder=True).prefetch(4)
    datasets[category] = (positive_ds, negative_ds)
  return datasets


# Use n-way one shot learning evaluation
def evaluate(model_path: str, records_path: str, categories: list, evals: int = 100, n_way=4) -> list:
  # Creates dataset to sample from
  datasets = create_test_dataset(model_path, records_path, categories, n_way)
  # Performs n-way one shot learning
  evaluations = []
  # Selects random classes to evaluate
  sampled_categories = np.random.choice(categories, evals, replace=True)
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
    distances = [
      (compute_squared_distance(one_shot_obs, tf.expand_dims(positive_obs, axis=0))[0], one_shot_category)
    ]
    # Compares the one shot embedding with n - 1 samples from drawn randomly from the other classes
    negative_obs, negative_labels = iter_negative.get_next()
    negative_distances = compute_squared_distance(one_shot_obs, negative_obs)
    for value, label in zip(negative_distances, negative_labels):
      distances.append((value, label.numpy()))

    entry = (one_shot_category, sorted(distances, key=lambda values: values[0]))
    evaluations.append(entry)
    end = time.perf_counter()
    print('Entry ({:.2f} secs):'.format(end - start), entry)
  # Computing matching estimations
  matching = [label == distances[0][1] for label, distances in evaluations]
  print('Closest matching target: {}'.format(sum(matching)))
  return evaluations


class AirSiameseLearner:
  """
  Siamese architecture to learn embeddings that allow to differentiate aircraft classes based on noise during take-off.
  """
  def __init__(self, regularize=True, batch_norm=False):
    # Stores options
    self.regularize = regularize
    self.batch_norm = batch_norm
    # Builds model architecture
    self.siamese, self.sibling = self.build_model()
    # Selects loss and optimizer
    self.loss = tf.keras.losses.BinaryCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam()

  # Builds siamese network and sibling network
  def build_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
    # Inputs to the siamese model. Data is fed as a pair (input1, input2)
    input1 = tf.keras.layers.Input((None, MFCC_SIZE, WINDOW_SIZE, 1), name='spec1')
    input2 = tf.keras.layers.Input((None, MFCC_SIZE, WINDOW_SIZE, 1), name='spec2')
    # Actual model receiving a spectrogram and outputing an embedding
    sibling = self.build_sibling_model()
    # Embeddings for both inputs
    embedding1 = sibling(input1)
    embedding2 = sibling(input2)
    # Computes similarity
    similarity = AbsoluteSimilarity()(embedding1, embedding2)
    # Dense layer to finally output probability of been the same or different class
    dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    output = dense(similarity)
    # Builds siamese architecture
    siamese = tf.keras.models.Model([input1, input2], output, name='siamese_nn')
    # Return tuple of models
    return siamese, sibling

  # Builds sibling model architecture returning a tf.keras.Model
  def build_sibling_model(self) -> tf.keras.Model:
    # Create inputs
    spectrogram = tf.keras.layers.Input((None, MFCC_SIZE, WINDOW_SIZE, 1))
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
    # Third convolutional layer. Timed distributed.
    conv3 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5,
                             padding='valid',
                             activation=tf.nn.relu,
                             data_format='channels_last'),
      name='Conv3'
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
    lstm1 = tf.keras.layers.LSTM(32, return_state=False, return_sequences=False)
    # Creates connections between layers of the model
    x = conv1(spectrogram)
    x = pooling1(x)
    if self.batch_norm:
      x = batch_norm1(x)
    x = conv2(x)
    x = conv3(x)
    x = pooling2(x)
    x = flatten1(x)
    if self.regularize:
      x = dropout1(x)
    if self.batch_norm:
      x = batch_norm2(x)
    embedding = lstm1(x)
    return tf.keras.Model(spectrogram, embedding, name='sibling_nn')

  # Prints out the model's summary
  def summary(self) -> None:
    self.siamese.summary()
    tf.keras.utils.plot_model(self.siamese, to_file='diagrams/siamese.jpg', show_shapes=True, expand_nested=True)

  @tf.function
  def train_step(self, batch):
    # Unwrapped inputs and expected output
    input1, input2, y_true = batch
    with tf.GradientTape() as tape:
      # Computes siamese architecture output
      y_pred = self.siamese([input1, input2])
      # Computes loss in batch (the <start> token should not be part of the expected output)
      loss = self.loss(y_true, tf.squeeze(y_pred))
    # Determines gradients
    variables = self.siamese.trainable_variables
    gradients = tape.gradient(loss, variables)
    # Updates params
    self.optimizer.apply_gradients(zip(gradients, variables))
    # Computes metric
    acc = tf.keras.metrics.binary_accuracy(tf.cast(y_true, tf.float32), tf.squeeze(y_pred))
    # Returns loss and metric
    return loss, acc

  # Trains model
  def fit(self, records_path: str, categories: list, epochs: int) -> None:
    # Creates train dataset using tf.data.Dataset
    train_ds = create_training_dataset(records_path, categories)
    # Creates an iterator to request batches
    iterator = iter(train_ds)
    # Training loop
    for epoch in range(epochs):
      start = time.perf_counter()
      # Gets a batch from dataset
      batch = iterator.get_next()
      # Performs update step
      loss, acc = self.train_step(batch)
      # Logs training results
      print('\nEpoch {} out of {} complete ({:.2f} secs) -- Batch Loss: {:.4f} -- Batch Acc: {:.2f}'.format(
        epoch + 1,
        epochs,
        time.perf_counter() - start,
        loss,
        tf.squeeze(acc)
      ), end='')


if __name__ == '__main__':
  # Selects CPU or GPU
  # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  # Training/Test data files
  train_file = '../exports/2020-03-01 07-34-19/train.tfrecord'
  test_file = '../exports/2020-03-01 07-34-19/test.tfrecord'

  # Performs training using only the first four classes
  learner = AirSiameseLearner(regularize=True, batch_norm=False)
  learner.fit(train_file, AIRCRAFT_EIGHT_LABELS, 600)
  learner.summary()

  # Saves trained sibling model
  date_string = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
  save_path = 'trained_model/air_siamense_sibling ' + date_string + '/'
  tf.keras.models.save_model(learner.sibling, save_path)

  # Evaluate trained model
  evaluations = evaluate(save_path, test_file, AIRCRAFT_EIGHT_LABELS)
