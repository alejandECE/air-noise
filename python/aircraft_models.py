#  Created by Luis Alejandro (alejand@umich.edu)
import tensorflow as tf


class AirBinaryTemporalCNN:
  """
      Simple CNN model with two 1D conv layers followed by global max pooling
      and a fully connected layer.

      Designed to perform binary classification on a simple dataset containing
      Airbus/Boeing aircraft take-off signals.
  """

  def __init__(self, use_regularizer=True):
    self.use_regularizer = use_regularizer
    self.model = None

  def build_model(self, input_shape):
    """
        Builds tf.keras model
    """
    # A 1D convolutional layer looking for timeline pattern in the original
    # spectrogram. Filters have 50% overlap in the time axis.
    conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=16,
                                   strides=8, padding='same',
                                   activation=tf.nn.relu)
    # Another 1D convolutional layer aftewards to generate more complex
    # time features. The kernel combines analyzes three consecutives
    # activation maps from the previous layer output.
    conv2 = tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu)
    # Performs global max pooling to "keep only the maximum" activation of
    # the previous convolutional layer filters. Technically answer where
    # (in time) the filter generated the strongest output.
    pooling1 = tf.keras.layers.GlobalMaxPooling1D()
    # Dense connecting layers to perform classification
    if self.use_regularizer:
      dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                     kernel_regularizer=
                                     tf.keras.regularizers.l2(0.1))
    else:
      dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    # Create connections and returns model
    inputs = tf.keras.Input(input_shape)
    x = conv1(inputs)
    x = conv2(x)
    x = pooling1(x)
    outputs = dense1(x)

    self.model = tf.keras.Model(inputs, outputs)

    return self.model


class AirMultinomialTemporalCNN:
  """
      Simple CNN model with two 1D conv layers followed by global max pooling
      and a fully connected layer.

      Desgined to perform multinomial classification on a simple dataset
      containing four classes of aircraft take-off signals.
  """

  def __init__(self, categories, use_regularizer=True):
    self.use_regularizer = use_regularizer
    self.categories = categories
    self.model = None

  def build_model(self, input_shape):
    """
        Builds tf.keras model
    """
    # A 1D convolutional layer looking for timeline pattern in the original
    # spectrogram. Filters have 50% overlap in the time axis.
    conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=16,
                                   strides=8, padding='same',
                                   activation=tf.nn.relu)
    # Another 1D convolutional layer aftewards to generate more complex
    # time features. The kernel combines analyzes three consecutives
    # activation maps from the previous layer output.
    conv2 = tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu)
    # Performs global max pooling to "keep only the maximum" activation of
    # the previous convolutional layer filters. Technically answer where
    # (in time) the filter generated the strongest output.
    pooling1 = tf.keras.layers.GlobalMaxPooling1D()
    # Dense connecting layers to perform classification
    if self.use_regularizer:
      dense1 = tf.keras.layers.Dense(self.categories, activation=tf.nn.softmax,
                                     kernel_regularizer=
                                     tf.keras.regularizers.l2(0.1))
    else:
      dense1 = tf.keras.layers.Dense(self.categories, activation=tf.nn.softmax)

    # Create connections and returns model
    inputs = tf.keras.Input(input_shape)
    x = conv1(inputs)
    x = conv2(x)
    x = pooling1(x)
    outputs = dense1(x)

    self.model = tf.keras.Model(inputs, outputs)

    return self.model


class AirBinaryRNN:

  def __init__(self, use_regularizer=True):
    self.use_regularizer = use_regularizer
    self.model = None

  def build_model(self, input_shape):
    pass
