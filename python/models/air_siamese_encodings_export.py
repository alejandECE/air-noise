#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf
from commons import AUTOTUNE
from air_siamese_architecture import parse_observation
from air_siamese_architecture import build_sequence
from air_siamese_architecture import get_embedding
import os

# Constants
BATCH_SIZE = 64

# Dictionary containing features description for parsing purposes
feature_description = {
  'embedding': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=[0.0]),
  'label': tf.io.FixedLenFeature([], tf.string, default_value='')
}


# Creates tf.data.Dataset with labeled embeddings observations from tfrecord files
def create_export_dataset(records_path: list, model_path: str) -> tf.data.Dataset:
  # Reads tfrecords from files
  records_ds = tf.data.TFRecordDataset(records_path)
  # Parses observation from tfrecords
  records_ds = records_ds.map(parse_observation, num_parallel_calls=AUTOTUNE)
  # Builds a sequence to feed siamese networks
  records_ds = records_ds.map(build_sequence, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
  # Get the embedding dataset using the model trained
  embedding_ds = records_ds.map(lambda inputs, labels: get_embedding(model_path, inputs, labels),
                                num_parallel_calls=AUTOTUNE).unbatch().prefetch(1)
  return embedding_ds


# Export embeddings to tfrecords
def export_embeddings_from_dataset(dataset: tf.data.Dataset, destination: str):
  with tf.io.TFRecordWriter(destination) as writer:
    for embedding, label in dataset:
      feature = {
        'embedding': tf.train.Feature(float_list=tf.train.FloatList(value=list(embedding.numpy().ravel()))),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.numpy()]))
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())


if __name__ == '__main__':
  # Selects CPU or GPU
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  # Training/Test data files
  train_file = '../exports/2020-03-01 07-34-19/train.tfrecord'
  test_file = '../exports/2020-03-01 07-34-19/test.tfrecord'

  # Model to generate the embedding with
  load_path = 'trained_model/air_siamense_sibling 2020-06-28 23-33-06'

  # Creates tf.data.Dataset with labeled embeddings observations
  ds = create_export_dataset([train_file, test_file], load_path)

  # Export observations to tfrecords
  save_path = 'embeddings/' + load_path.split(sep='/')[-1] + '.tfrecord'
  export_embeddings_from_dataset(ds, save_path)
