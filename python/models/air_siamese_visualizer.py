#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import tensorflow as tf
from commons import AUTOTUNE
from commons import AIRCRAFT_EIGHT_LABELS
from air_siamese_encodings_export import feature_description
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

# Constants
BATCH_SIZE = 64


# Parses embedding from tfrecord example
@tf.function
def parse_observation(example: tf.Tensor) -> Tuple:
  observation = tf.io.parse_single_example(example, feature_description)
  return observation['embedding'], observation['label']


# Only returns true for the listed categories
@tf.function
def filter_categories(label: tf.Tensor, categories: list) -> tf.Tensor:
  return tf.reduce_any(tf.equal(label, categories))


# Creates dataset with observation from the listed categories
def create_embeddings_dataset(records_path: str, categories: list) -> tf.data.Dataset:
  records_ds = tf.data.TFRecordDataset([records_path])
  records_ds = records_ds.map(parse_observation, num_parallel_calls=AUTOTUNE)
  records_ds = records_ds.filter(lambda embedding, label: filter_categories(label, categories))
  records_ds = records_ds.batch(BATCH_SIZE).prefetch(1)
  return records_ds


# Finds PCA using incremental approach
def find_incremental_pca(dataset: tf.data.Dataset) -> IncrementalPCA:
  pca = IncrementalPCA(n_components=2)
  for batch in dataset:
    pca.fit(batch[0].numpy())
  return pca


def plot_embeddings(dataset: tf.data.Dataset, categories: list):
  # Finds PCA
  pca = find_incremental_pca(dataset)
  # Creates colomap and supporting dictionary
  viridis = cm.get_cmap('viridis', len(categories))
  colors = {category: viridis(index / len(categories)) for index, category in enumerate(categories)}
  fig = plt.figure()
  fig.set_size_inches(10, 6)
  ax = fig.add_subplot()
  # Goes through every batch adding scatter plots
  for embeddings, labels in dataset:
    reduced_embeddings = pca.transform(embeddings.numpy())
    ax.scatter(reduced_embeddings[:, 0],
               reduced_embeddings[:, 1],
               c=[colors[label] for label in labels.numpy()])
  # Creates legend
  plt.legend(handles=[
    mpatches.Patch(color=colors[category], label=category) for category in categories
  ])
  plt.show()


if __name__ == '__main__':
  # Embeddings file
  embeddings_path = 'embeddings/air_siamense_sibling 2020-06-28 23-33-06.tfrecord'
  # Classes to plot
  classes = AIRCRAFT_EIGHT_LABELS
  # Loads embeddings
  embedding_ds = create_embeddings_dataset(embeddings_path, classes)
  # Plots dataset
  plot_embeddings(embedding_ds, classes)
