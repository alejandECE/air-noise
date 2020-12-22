#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import datetime
from collections import namedtuple
import tensorflow as tf
import pathlib
import os

# Constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
EXPERIMENTS_FOLDER = 'experiments'
LOCAL_DIAGRAM_FOLDER = 'diagrams'
LOCAL_TRAINED_MODEL_FOLDER = 'trained_model'
LOCAL_EMBEDDING_FOLDER = 'embeddings'

# To keep track of classes in folder
Entry = namedtuple('Entry', ['label', 'samples'])


# Get classes/categories/labels from directory sorted with the most common one first
def get_classes_from_folder(folder: pathlib.Path) -> list:
  classes = []
  for root, _, files in os.walk(folder):
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
def get_tfrecords_from_folder(folder: pathlib.Path) -> list:
  records = []
  for element in os.listdir(folder):
    if '.tfrecord' in element and 'embeddings' not in element:
      records.append(str(pathlib.Path(folder) / element))
  return records


# Verify the tfrecord files exist in the directory
def verify_default_records_from_folder(folder: pathlib.Path, records_names: list = None) -> None:
  if records_names is None:
    records_names = ['train.tfrecord', 'test.tfrecord']
  msg = 'The records do not exist in the folder specified '
  assert all([(folder / name).exists() for name in records_names]), msg


# Returns a trained model from the given the experiment folder
def get_model_from_experiment(experiment_folder: pathlib.Path, model_name: str) -> tf.keras.Model:
  return tf.keras.models.load_model(str(experiment_folder / LOCAL_TRAINED_MODEL_FOLDER / model_name))


# Generates an experiment folder using current timestamp for the given dataset folder
def generate_experiment_path(dataset_folder: pathlib.Path) -> pathlib.Path:
  path = dataset_folder / EXPERIMENTS_FOLDER / datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  if not path.exists():
    path.mkdir(parents=True)
  return path


# Generates the path of the model for the given experiment folder
def generate_model_path(experiment_folder: pathlib.Path, model_name: str) -> pathlib.Path:
  path = experiment_folder / LOCAL_TRAINED_MODEL_FOLDER / model_name
  if not path.exists():
    path.mkdir(parents=True)
  return path


# Generates diagram folder for the given experiment folder
def generate_diagram_path(experiment_folder: pathlib.Path, model_name: str) -> pathlib.Path:
  path = experiment_folder / LOCAL_DIAGRAM_FOLDER
  if not path.exists():
    path.mkdir(parents=True)
  return path / (model_name + '.jpg')


# Generates embedding folder using current timestamp for the given experiment folder
def generate_embeddings_path(experiment_folder: pathlib.Path) -> pathlib.Path:
  path = experiment_folder / LOCAL_EMBEDDING_FOLDER / datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  if not path.exists():
    path.mkdir(parents=True)
  return path / 'embeddings.tfrecord'


def generate_tensorboard_path(embedding_path: pathlib.Path) -> pathlib.Path:
  path = embedding_path.parent / 'tensorboard'
  if not path.exists():
    path.mkdir(parents=True)
  return path


# Returns experiment folder for a given embeddings folder
def get_experiment_from_embeddings(embeddings_path: pathlib.Path) -> pathlib.Path:
  return embeddings_path.parent.parent.parent


# Returns dataset folder for a given experiment folder
def get_dataset_from_experiment(experiment_path: pathlib.Path) -> pathlib.Path:
  return experiment_path.parent.parent


def get_classes_from_file(file: pathlib.Path) -> list:
  with open(file, 'r') as f:
    return [line[:-1].encode('utf8') for line in f.readlines()]


if __name__ == '__main__':
  directory = '../exports/2020-03-01 07-34-19 (eight classes)/'
  labels = get_classes_from_folder(pathlib.Path(directory))
  print(labels)
  verify_default_records_from_folder(pathlib.Path(directory))
  files = get_tfrecords_from_folder(directory)
  print(files)
