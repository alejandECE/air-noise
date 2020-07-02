#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import pathlib
import os

# Constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
EXPERIMENTS_FOLDER = 'experiments'
LOCAL_DIAGRAM_FOLDER = 'diagrams'
LOCAL_TRAINED_MODEL_FOLDER = 'trained_model'


# Get classes/categories/labels from directory
def get_classes_from_directory(directory: str) -> list:
  classes = []
  for root, _, files in os.walk(directory):
    for file in files:
      # A folder is considered a category if contains any npy file
      if '.npy' in file:
        classes.append(root.split('\\')[-1].encode('utf8'))
        break
  return classes


# Verify the tfrecord files exist in the directory
def verify_tfrecords_from_directory(directory: str, records_names: list = None) -> list:
  if records_names is None:
    records_names = ['train.tfrecord', 'test.tfrecord']
  return all([(pathlib.Path(directory) / name).exists() for name in records_names])


if __name__ == '__main__':
  folder = '../exports/2020-02-07 01-09-35/'
  labels = get_classes_from_directory(folder)
  print(labels)
  exists = verify_tfrecords_from_directory(folder)
  print(exists)

