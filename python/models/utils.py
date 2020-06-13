#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

from typing import Tuple
import tensorflow as tf
import numpy as np


# Computes accuracy and macro averaged metrics: precision, recall and fscore
def get_macro_metrics(matrix: np.array, beta=1) -> Tuple:
  tp = np.diag(matrix)
  filtered_matrix = np.multiply(1. - np.eye(matrix.shape[0]), matrix)
  fp = filtered_matrix.sum(axis=0)
  fn = filtered_matrix.sum(axis=1)
  acc = tp.sum() / matrix.sum()
  recall = (tp / (tp + fn)).mean()
  precision = (tp / (tp + fp)).mean()
  fscore = (1 + beta ** 2) * (precision * recall) / (((beta ** 2) * precision) + recall)

  return acc, precision, recall, fscore


# Determines the confusion matrix of the classification model
def confusion_matrix(model: tf.keras.Model, dataset: tf.data.Dataset):
  # Determining the number of classes
  categories = model.layers[-1].variables[0].shape[1]
  if categories == 1:
    categories = 2

  # Confusion matrix tensor to be returned
  matrix = tf.zeros((categories, categories), dtype=tf.int64)

  # Goes trough every batch in the dataset
  for x, y in dataset:
    # Indexes represents entries in the confusion matrix
    y_true = tf.cast(tf.squeeze(y), dtype=tf.int64)
    if categories == 2:
      y_pred = tf.cast(tf.squeeze(model.predict(x) >= 0.5), dtype=tf.int64)
    else:
      y_pred = tf.cast(tf.squeeze(tf.argmax(model.predict(x), axis=1)), dtype=tf.int64)
    indexes = tf.stack((y_true, y_pred), axis=1)

    # Creates sparse vector to represent matrix. This matrix representation will have many repeated indexes with a value
    # of one. Based on this repetition is that each entry in the matrix is generated.
    counts = tf.SparseTensor(indexes,
                             values=tf.ones_like(y_true, dtype=tf.int64),
                             dense_shape=(categories, categories))
    # Hack to convert the sparse representation to a dense confusion matrix (for current batch), where values for
    # repeated indexes will be added together.
    update = tf.sparse.add(tf.zeros((categories, categories), dtype=tf.int64), counts)

    # Updates
    matrix += update

  return matrix


# Shows performance after training displaying accuracy and confusion matrix
def display_performance(model: tf.keras.Model,
                        train_ds: tf.data.Dataset,
                        test_ds: tf.data.Dataset) -> None:

  print('\n------- Training --------')
  matrix = confusion_matrix(model, train_ds).numpy()
  print('Confusion Matrix:')
  print(matrix)
  accuracy, precision, recall, fscore = get_macro_metrics(matrix)
  print('\nAccuracy: {:.4f}'.format(accuracy))
  print('Precision: {:.4f}'.format(precision))
  print('Recall: {:.4f}'.format(recall))
  print('F1 Score: {:.4f}'.format(fscore))

  print('\n--------- Test ----------')
  matrix = confusion_matrix(model, test_ds).numpy()
  print('Confusion Matrix:')
  print(matrix)
  accuracy, precision, recall, fscore = get_macro_metrics(matrix)
  print('\nAccuracy: {:.4f}'.format(accuracy))
  print('Precision: {:.4f}'.format(precision))
  print('Recall: {:.4f}'.format(recall))
  print('F1 Score: {:.4f}'.format(fscore))
