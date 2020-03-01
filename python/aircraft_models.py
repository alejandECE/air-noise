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
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.1)
    # Dense connecting layers to perform classification
    if self.use_regularizer:
      dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                     kernel_regularizer=
                                     tf.keras.regularizers.l2(0.3))
    else:
      dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    # Create connections and returns model
    inputs = tf.keras.Input(input_shape)
    x = conv1(inputs)
    x = conv2(x)
    x = pooling1(x)
    if self.use_regularizer:
      x = dropout1(x)
    outputs = dense1(x)

    self.model = tf.keras.Model(inputs, outputs)

    return self.model


class AirMultinomialTemporalCNN:
  """
  Simple CNN model with two 1D conv layers followed by global max pooling
  and a fully connected layer.

  Designed to perform multinomial classification on a simple dataset
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
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.1)
    # Dense connecting layers to perform classification
    if self.use_regularizer:
      dense1 = tf.keras.layers.Dense(self.categories, activation=tf.nn.softmax,
                                     kernel_regularizer=
                                     tf.keras.regularizers.l2(0.3))
    else:
      dense1 = tf.keras.layers.Dense(self.categories, activation=tf.nn.softmax)

    # Create connections and returns model
    inputs = tf.keras.Input(input_shape)
    x = conv1(inputs)
    x = conv2(x)
    x = pooling1(x)
    if self.use_regularizer:
      x = dropout1(x)
    outputs = dense1(x)

    self.model = tf.keras.Model(inputs, outputs)
    return self.model


class SpecSequence(tf.keras.layers.Layer):
  """
  A non-trainable layer to generate a sequence of potentially overlapping window from an input spectrogram.
  Inputs are expected to be (batch, freq, time)
  Outputs will be (batch, windows, freq, time)
  """
  def __init__(self, window_size: int, window_overlap=0.5, name='Sequencer', **kwargs):
    super(SpecSequence, self).__init__(name=name, **kwargs)
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


class AirBinaryRNN:
  """
  Simple recurrent neural network model using a sequence of spectrogram windows. The input of this model
  will be a sequence of overlapping spectrogram slices (windows).

  Designed to perform binary classification on a simple dataset containing
  Airbus/Boeing aircraft take-off signals.
  """

  def __init__(self, use_regularizer=True, use_sequencer=False, window_size=50, window_overlap=0.5):
    self.use_regularizer = use_regularizer
    self.use_sequencer = use_sequencer
    self.window_size = window_size
    self.window_overlap = window_overlap
    self.model = None

  def build_model(self, input_shape):
    # Creates the sequencer layer if requested
    if self.use_sequencer:
      sequencer1 = SpecSequence(self.window_size, self.window_overlap)
    # First convolutional layer to find spatial features in a segment of the spectrogram. Timed distributed to get
    # applied to every segment of the inputted spectrogram.
    conv1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5, padding='valid', activation=tf.nn.relu, data_format='channels_last'),
      name='Conv1'
    )
    # Pooling
    pooling1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3),
      name='Pool1'
    )
    # Second convolutional layer. Timed distributed to get applied to every segment of the inputted spectrogram.
    conv2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5, padding='valid', activation=tf.nn.relu, data_format='channels_last'),
      name='Conv2'
    )
    # Pooling
    pooling2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3),
      name='Pool2'
    )
    # Flatten results from convolutions/pooling to be fed to the lstm layers
    flatten1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Flatten(),
      name='Flatten'
    )
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.3)
    # Recurrent layer to capture temporal relationships
    lstm1 = tf.keras.layers.LSTM(32, return_state=False, return_sequences=False)
    # Dropout layer
    dropout2 = tf.keras.layers.Dropout(0.1)
    # Dense to make the final classification
    if self.use_regularizer:
      dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
                                     kernel_regularizer=
                                     tf.keras.regularizers.l2(0.3),
                                     name='Dense')
    else:
      dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name='Dense')

    # Create connections and returns model
    inputs = tf.keras.layers.Input(input_shape)
    if self.use_sequencer:
      x = sequencer1(inputs)
      x = conv1(x)
    else:
      x = conv1(inputs)
    x = pooling1(x)
    x = conv2(x)
    x = pooling2(x)
    x = flatten1(x)
    if self.use_regularizer:
      x = dropout1(x)
    x = lstm1(x)
    if self.use_regularizer:
      x = dropout2(x)
    outputs = dense1(x)

    self.model = tf.keras.Model(inputs, outputs)
    return self.model


class AirMultinomialRNN:
  """
  Simple recurrent neural network model using a sequence of spectrogram windows. The input of this model
  will be a sequence of overlapping spectrogram slices (windows).

  Designed to perform multinomial classification on a simple dataset
  containing four classes of aircraft take-off signals.
  """

  def __init__(self, categories, use_regularizer=True, use_sequencer=False, window_size=50, window_overlap=0.5):
    self.categories = categories
    self.use_regularizer = use_regularizer
    self.use_sequencer = use_sequencer
    self.window_size = window_size
    self.window_overlap = window_overlap
    self.model = None

  def build_model(self, input_shape):
    # Creates the sequencer layer if requested
    if self.use_sequencer:
      sequencer1 = SpecSequence(self.window_size, self.window_overlap)
    # First convolutional layer to find spatial features in a segment of the spectrogram. Timed distributed to get
    # applied to every segment of the inputted spectrogram.
    conv1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5, padding='valid', activation=tf.nn.relu, data_format='channels_last'),
      name='Conv1'
    )
    # Pooling
    pooling1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3),
      name='Pool1'
    )
    # Second convolutional layer. Timed distributed to get applied to every segment of the inputted spectrogram.
    conv2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Conv2D(32, 5, padding='valid', activation=tf.nn.relu, data_format='channels_last'),
      name='Conv2'
    )
    # Pooling
    pooling2 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPool2D(3),
      name='Pool2'
    )
    # Flatten results from convolutions/pooling to be fed to the lstm layers
    flatten1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.Flatten(),
      name='Flatten'
    )
    # Dropout layer
    dropout1 = tf.keras.layers.Dropout(0.4)
    # Recurrent layer to capture temporal relationships
    lstm1 = tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(32, return_state=False, return_sequences=True),
      name='Bidirectional'
    )
    # Global max pool
    global1 = tf.keras.layers.GlobalMaxPool1D()
    # Dense to make the final classification
    if self.use_regularizer:
      dense1 = tf.keras.layers.Dense(self.categories, activation=tf.nn.softmax,
                                     kernel_regularizer=
                                     tf.keras.regularizers.l2(0.2),
                                     name='Dense')
    else:
      dense1 = tf.keras.layers.Dense(self.categories, activation=tf.nn.softmax, name='Dense')

    # Create connections and returns model
    inputs = tf.keras.layers.Input(input_shape)
    if self.use_sequencer:
      x = sequencer1(inputs)
      x = conv1(x)
    else:
      x = conv1(inputs)
    x = pooling1(x)
    x = conv2(x)
    x = pooling2(x)
    x = flatten1(x)
    if self.use_regularizer:
      x = dropout1(x)
    x = lstm1(x)
    x = global1(x)
    outputs = dense1(x)

    self.model = tf.keras.Model(inputs, outputs)
    return self.model
