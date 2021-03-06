#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import os
import re
import random
import numpy as np
import tensorflow as tf

# Dictionary containing features description for parsing purposes
feature_description = {
  'spec': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=[0.0]),
  'mfcc': tf.io.FixedLenFeature([], tf.int64),
  'samples': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
  'measurement': tf.io.FixedLenFeature([], tf.string, default_value=''),
  'array': tf.io.FixedLenFeature([], tf.string, default_value=''),
  'sensor': tf.io.FixedLenFeature([], tf.string, default_value='')
}


def serialize_observation(observation):
  """
  Serializes a single observation
  """
  # Read npy file
  url, label, measurement, array, sensor = observation
  spectrogram = np.load(url)
  mfcc, samples = spectrogram.shape
  # Create a dictionary mapping the feature name to the tf.Example compatible data type
  feature = {
    'spec': tf.train.Feature(float_list=tf.train.FloatList(value=list(spectrogram.ravel()))),
    'mfcc': tf.train.Feature(int64_list=tf.train.Int64List(value=[mfcc])),
    'samples': tf.train.Feature(int64_list=tf.train.Int64List(value=[samples])),
    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode()])),
    'measurement': tf.train.Feature(bytes_list=tf.train.BytesList(value=[measurement.encode()])),
    'array': tf.train.Feature(bytes_list=tf.train.BytesList(value=[array.encode()])),
    'sensor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sensor.encode()]))
  }
  # Create a Features message using tf.train.Example
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()


class AircraftRecordBuilder(object):
  """
  Creates two records files (training/test)
  It ensures the percentage of test samples of each class is the same.
  For the training set it includes all signals from a measurement but for
  the test set only one random signal from each measurement is included.
  """

  def __init__(self, path):
    # Stores path
    self.path = path
    # Reads all npy files found inside path
    self.datafiles = []
    for root, _, files in os.walk(path):
      for file in files:
        if '.npy' in file:
          groups = re.match(r'^m(\d+)a(\d+)s(\d+)', file).groups()
          url = os.path.join(root, file)
          label = os.path.split(root)[1]
          measurement = groups[0]
          array = groups[1]
          sensor = groups[2]
          # appending (url, class name, measurement id, array, sensor)
          self.datafiles.append((url, label, measurement, array, sensor))
    random.shuffle(self.datafiles)

  def build(self, test_pct=0.2):
    """
    Generates a separate tfrecord file containing the serialized observations
    for each stratified set (training & test) generated from the signals
    found in the input path.
    """
    # generate sets of files
    train_set, test_set = self.generate_sets(test_pct)
    # generate a tfrecord for each set
    self.generate_tfrecord(train_set, 'train.tfrecord')
    self.generate_tfrecord(test_set, 'test.tfrecord')

  def generate_sets(self, test_pct):
    """
    Creates stratified training and test sets of the signals found in the
    input path
    """
    # Shuffles dataset
    measurements = []
    labels = []
    for _, label, measurement, _, _ in self.datafiles:
      measurements.append(measurement)
      labels.append(label)
    measurements = np.array(measurements)
    labels = np.array(labels)
    classes = np.unique(labels)
    # Splits dataset
    m = len(self.datafiles)
    train_obs = np.zeros(m, dtype=bool)
    test_obs = np.zeros(m, dtype=bool)
    for category in classes:
      indexes = np.unique(measurements[labels == category])
      indexes = np.random.permutation(indexes)
      k = int(len(indexes) * (1 - test_pct))
      # Training set
      for index in indexes[:k]:
        new_obs = (measurements == index)
        train_obs = train_obs | new_obs
      # Test set
      for index in indexes[k:]:
        new_obs = (measurements == index)
        mask = np.random.permutation(new_obs.nonzero()[0])
        new_obs[mask[:-1]] = False
        test_obs = test_obs | new_obs

    train_set = [self.datafiles[i] for i in range(len(train_obs)) if train_obs[i]]
    test_set = [self.datafiles[i] for i in range(len(test_obs)) if test_obs[i]]

    return train_set, test_set

  def generate_tfrecord(self, observations, filename):
    """
    Generates a tfrecord file containing the serialized observations
    """
    filepath = os.path.join(self.path, filename)
    with tf.io.TFRecordWriter(filepath) as writer:
      for obs in observations:
        example = serialize_observation(obs)
        writer.write(example)
