#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import os
import argparse
import pathlib
from collections import Counter
import tensorflow as tf
from air_siamese_architecture import parse_observation
from air_siamese_architecture import get_embedding
from air_siamese_architecture import SIBLING_MODEL_NAME
import commons

# Constants
BATCH_SIZE = 64

# Dictionary containing features description for parsing purposes
feature_description = {
  'embedding': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=[0.0]),
  'label': tf.io.FixedLenFeature([], tf.string, default_value='')
}


# Creates tf.data.Dataset with labeled embeddings observations from tfrecord files
def create_export_dataset(tfrecords: list, sibling: tf.keras.Model) -> tf.data.Dataset:
  # Reads tfrecords from files
  records_ds = tf.data.TFRecordDataset(tfrecords)
  # Parses observation from tfrecords
  records_ds = records_ds.map(parse_observation, num_parallel_calls=commons.AUTOTUNE)
  # Batches dataset
  records_ds = records_ds.batch(BATCH_SIZE)
  # Get the embedding dataset using the model trained
  embedding_ds = records_ds.map(lambda inputs, labels: get_embedding(sibling, inputs, labels),
                                num_parallel_calls=commons.AUTOTUNE).unbatch().prefetch(1)
  return embedding_ds


# Export embeddings to tfrecords
def export_embeddings_from_dataset(dataset: tf.data.Dataset, embeddings_path: pathlib.Path):
  # Generates embeddings and stores them in a tfrecord file
  classes_counter = Counter()
  with tf.io.TFRecordWriter(str(embeddings_path)) as writer:
    for embedding, label in dataset:
      classes_counter[label.numpy()] += 1
      feature = {
        'embedding': tf.train.Feature(float_list=tf.train.FloatList(value=list(embedding.numpy().ravel()))),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.numpy()]))
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
  # Stores labels in the current dataset
  classes_path = embeddings_path.parent / 'classes.txt'
  labels = [item[0] for item in sorted(classes_counter.items(), key=lambda item: item[1], reverse=True)]
  with open(classes_path, 'w') as file:
    file.writelines([entry.decode() + '\n' for entry in labels])


if __name__ == '__main__':
  # Selects CPU or GPU
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  # Parsing arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('folder', help='Experiment folder containing results', type=str)
  args = parser.parse_args()
  folder = args.folder
  # Creates tf.data.Dataset with labeled embeddings observations
  experiment_path = pathlib.Path(folder)
  model = commons.get_model_from_experiment(experiment_path, SIBLING_MODEL_NAME)
  ds = create_export_dataset(commons.get_tfrecords_from_folder(commons.get_dataset_from_experiment(experiment_path)),
                             model)
  # Export observations to tfrecords
  export_embeddings_from_dataset(ds, commons.generate_embeddings_path(experiment_path))
