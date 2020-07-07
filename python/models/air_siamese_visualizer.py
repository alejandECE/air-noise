#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import argparse
import pathlib
from typing import Tuple
import tensorflow as tf
from tensorboard.plugins import projector
from commons import AUTOTUNE
from commons import get_classes_from_file
from commons import generate_tensorboard_path
from air_siamese_encodings_export import feature_description
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

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
  if len(categories) == 0:
    return True
  return tf.reduce_any(tf.equal(label, categories))


# Creates dataset with observation from the listed categories
def create_embeddings_dataset(records_path: str, categories: list) -> tf.data.Dataset:
  records_ds = tf.data.TFRecordDataset([records_path])
  records_ds = records_ds.map(parse_observation, num_parallel_calls=AUTOTUNE)
  records_ds = records_ds.filter(lambda embedding, label: filter_categories(label, categories))
  records_ds = records_ds.batch(BATCH_SIZE).prefetch(1)
  return records_ds


# Finds PCA using incremental approach
def find_incremental_pca(dataset: tf.data.Dataset, components: int) -> IncrementalPCA:
  pca = IncrementalPCA(n_components=components)
  for batch in dataset:
    pca.fit(batch[0].numpy())
  return pca


def find_pca(dataset: tf.data.Dataset, components: int) -> PCA:
  pca = PCA(n_components=components)
  batches = []
  for batch in dataset:
    batches.append(batch[0].numpy())
  X = np.vstack(batches)
  pca.fit(X)
  return pca


# Plots embeddings in 3D
def plot_embeddings_3d(dataset: tf.data.Dataset, categories: list, method='normal') -> None:
  # Finds PCA
  if method == 'normal':
    pca = find_pca(dataset, 3)
  elif method == 'incremental':
    pca = find_incremental_pca(dataset, 3)
  # Creates colomap and supporting dictionary
  viridis = cm.get_cmap('viridis', len(categories))
  colors = {category: viridis(index / len(categories)) for index, category in enumerate(categories)}
  # Creates figure
  fig = plt.figure()
  fig.set_size_inches(10, 6)
  ax = fig.add_subplot(111, projection='3d')
  # Goes through every batch adding scatter plots
  for embeddings, labels in dataset:
    reduced_embeddings = pca.transform(embeddings.numpy())
    ax.scatter(reduced_embeddings[:, 0],
               reduced_embeddings[:, 1],
               reduced_embeddings[:, 2],
               c=[colors[label] for label in labels.numpy()])
  # Create titles and labels
  ax.set_title('PCA Embeddings (Total Var: {:.2f})'.format(pca.explained_variance_ratio_.sum()))
  ax.set_xlabel('First PC (Var: {:.2f})'.format(pca.explained_variance_ratio_[0]))
  ax.set_ylabel('Second PC (Var: {:.2f})'.format(pca.explained_variance_ratio_[1]))
  ax.set_zlabel('Third PC (Var: {:.2f})'.format(pca.explained_variance_ratio_[2]))
  ax.view_init(40, -60)
  # Creates legend
  plt.legend(handles=[
    mpatches.Patch(color=colors[category], label=category) for category in categories
  ])
  plt.show()


# Plots embeddings in 2D
def plot_embeddings_2d(dataset: tf.data.Dataset, categories: list, method='normal') -> None:
  # Finds PCA
  if method == 'normal':
    pca = find_pca(dataset, 3)
  elif method == 'incremental':
    pca = find_incremental_pca(dataset, 3)
  # Creates colomap and supporting dictionary
  viridis = cm.get_cmap('viridis', len(categories))
  colors = {category: viridis(index / len(categories)) for index, category in enumerate(categories)}
  # Creates figure
  fig = plt.figure()
  fig.set_size_inches(10, 6)
  ax = fig.add_subplot()
  # Goes through every batch adding scatter plots
  for embeddings, labels in dataset:
    reduced_embeddings = pca.transform(embeddings.numpy())
    ax.scatter(reduced_embeddings[:, 0],
               reduced_embeddings[:, 1],
               c=[colors[label] for label in labels.numpy()])
  # Create titles and labels
  ax.set_title('PCA Embeddings (Total Var: {:.2f})'.format(pca.explained_variance_ratio_.sum()))
  ax.set_xlabel('First PC (Var: {:.2f})'.format(pca.explained_variance_ratio_[0]))
  ax.set_ylabel('Second PC (Var: {:.2f})'.format(pca.explained_variance_ratio_[1]))
  # Creates legend
  plt.legend(handles=[
    mpatches.Patch(color=colors[category], label=category) for category in categories
  ])
  plt.show()


# Exports embeddings to Tensorboard
def export_to_tensorboard(dataset: tf.data.Dataset, embedding_path: pathlib.Path):
  # Set up a logs directory, so Tensorboard knows where to look for files
  log_path = generate_tensorboard_path(embedding_path)
  # Creates embedding checkpoint with data
  with open(log_path / 'metadata.tsv', "w") as f:
    embeddings = []
    for batch_embeddings, batch_labels in dataset:
      embeddings.append(batch_embeddings)
      for label in batch_labels:
        f.write("{}\n".format(label))
    checkpoint = tf.train.Checkpoint(embeddings=tf.Variable(tf.concat(embeddings, axis=0)))
    checkpoint.save(str(log_path / "embeddings.ckpt"))
  # Set up config
  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()
  # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
  embedding.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
  embedding.metadata_path = 'metadata.tsv'
  projector.visualize_embeddings(str(log_path), config)


if __name__ == '__main__':
  # Parsing arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('folder', help='Embedding folder containing a embeddings.tfrecord file', type=str)
  parser.add_argument('classes', help='Text file name with listed classes to include', type=str)
  # Embeddings file
  args = parser.parse_args()
  folder = pathlib.Path(args.folder)
  embeddings_path = folder / 'embeddings.tfrecord'
  classes_path = folder / args.classes
  # Classes to plot
  classes = get_classes_from_file(classes_path)
  # Loads embeddings
  embedding_ds = create_embeddings_dataset(str(embeddings_path), classes)
  # Plots dataset 2d
  plot_embeddings_2d(embedding_ds, classes)
  plot_embeddings_2d(embedding_ds, classes, method='incremental')
  # Plots dataset 3d
  plot_embeddings_3d(embedding_ds, classes)
  plot_embeddings_3d(embedding_ds, classes, method='incremental')
  # Exports to Tensorboard
  export_to_tensorboard(embedding_ds, embeddings_path)
