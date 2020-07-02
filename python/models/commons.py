#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
import os

# Most frequent eight classes used from dataset
AIRCRAFT_EIGHT_LABELS = [
  b'A320-2xx (CFM56-5)', b'B737-7xx (CF56-7B22-)', b'ERJ190 (CF34-10E)', b'B737-8xx (CF56-7B22+)',
  b'ERJ145 (AE3007)', b'A320-2xx (V25xx)', b'A319-1xx (V25xx)', b'ERJ170-175 (CF34-8E)'
]

# Autotune constant from tf
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Get classes/categories/labels from directory
def get_classes_from_directory(directory: str) -> list:
  classes = []
  for root, _, files in os.walk(directory):
    for file in files:
      # A folder is considered a category if contains any npy file
      if '.npy' in file:
        classes.append(root.split('/')[-1].encode('utf8'))
        break
  return classes


# Verify the tfrecord files exist in the directory
def verify_tfrecords_from_directory(directory: str, records_names: list = None) -> list:
  if records_names is None:
    records_names = ['train.tfrecord', 'test.tfrecord']
  return all([os.path.exists(os.path.join(directory, name)) for name in records_names])


if __name__ == '__main__':
  folder = '../exports/2020-02-07 01-09-35/'
  labels = get_classes_from_directory(folder)
  print(labels)
  exists = verify_tfrecords_from_directory(folder)
  print(exists)

