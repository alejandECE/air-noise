#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf


class SpecSequencer(tf.keras.layers.Layer):
  """
  A non-trainable layer to generate a sequence of potentially overlapping window from an input spectrogram.

  Inputs are expected to be (batch, freq, time)
  Outputs will be (batch, windows, freq, time)
  """

  def __init__(self, window_size: int, window_overlap=0.5, name='Sequencer', **kwargs):
    super(SpecSequencer, self).__init__(name=name, **kwargs)
    # Config to determine sequences
    self.window_size = window_size
    self.window_overlap = window_overlap
    # Items related to the input/output shape
    self.indexes = None
    self.channels = None
    self.mfcc = None
    self.length = None
    # Not trainable layer
    self.trainable = False

  def build(self, input_shape):
    # Determines indexes for slicing
    samples = input_shape[2]
    step = int(self.window_size * (1.0 - self.window_overlap))
    start = tf.range(0, samples - self.window_size, step)
    end = tf.range(self.window_size, samples, step)
    self.indexes = tf.stack([start, end], axis=1)
    self.length = start.shape[0]
    self.channels = input_shape[-1]
    self.mfcc = input_shape[0]

  def call(self, inputs):
    # Check slices indexes are generated already
    if self.indexes is None:
      raise Exception('You must build the layer before calling it.')
    # Get slices from input
    sequence = tf.map_fn(lambda index: inputs[:, :, index[0]:index[1], :],
                         self.indexes,
                         back_prop=False,
                         dtype=tf.float32)
    # Set proper output shape
    output = tf.transpose(sequence, perm=[1, 0, 2, 3, 4])
    shape = tf.TensorShape(
      [None, self.length, self.mfcc, self.window_size, self.channels]
    )
    output.set_shape(shape)
    return output
