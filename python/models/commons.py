#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from collections import namedtuple
import tensorflow as tf
import pathlib
import os

# Constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
EXPERIMENTS_FOLDER = 'experiments'
LOCAL_DIAGRAM_FOLDER = 'diagrams'
LOCAL_TRAINED_MODEL_FOLDER = 'trained_model'

# To keep track of classes in folder
Entry = namedtuple('Entry', ['label', 'samples'])


# Get classes/categories/labels from directory sorted with the most common one first
def get_classes_from_directory(directory: pathlib.Path) -> list:
  classes = []
  for root, _, files in os.walk(directory):
    for file in files:
      # A folder is considered a category if contains any npy file
      if '.npy' in file:
        classes.append(Entry(
          label=root.split('\\')[-1].encode('utf8'),
          samples=len(files)
        ))
        break
  return [entry.label for entry in sorted(classes, key=lambda entry: entry.samples, reverse=True)]


# Get tfrecords files in folder
def get_tfrecords_from_directory(directory: pathlib.Path) -> list:
  records = []
  for root, _, files in os.walk(directory):
    for file in files:
      if '.tfrecord' in file:
        records.append(str(pathlib.Path(root) / file))
  return records


# Verify the tfrecord files exist in the directory
def verify_tfrecords_from_directory(directory: pathlib.Path, records_names: list = None) -> bool:
  if records_names is None:
    records_names = ['train.tfrecord', 'test.tfrecord']
  return all([(directory / name).exists() for name in records_names])


if __name__ == '__main__':
  folder = '../exports/2020-02-07 01-09-35/'
  labels = get_classes_from_directory(folder)
  print(labels)
  exists = verify_tfrecords_from_directory(folder)
  print(exists)

