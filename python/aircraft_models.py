import tensorflow as tf

class AirBinaryTemporalCNN():
    '''
        Simple CNN model with two 1D conv layers followed by global max pooling
        and a fully connected layer. 
        
        Desgined to perform binary classificaton on a simple dataset containing
        Airbus/Boeing aircraft take-off signals.
    '''
    
    # Constant values
    FEATURE_DESCRIPTION = {
        'spec': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True,default_value=[0.0]),
        'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'measurement': tf.io.FixedLenFeature([],tf.string, default_value=''),
        'array': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'sensor': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    NUMBER_MFCC = 128
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    def __init__(self, use_regularizer=True):
        self.use_regularizer = use_regularizer
        self.model = None
    
    '''
        Builds tf.keras model
    '''
    def build_model(self):
        # A 1D convolutional layer looking for timeline pattern in the original
        # spectrogram. Filters have 50% overlap in the time axis.
        conv1 = tf.keras.layers.Conv1D(filters=16,kernel_size=16,
                                            strides=8,padding='same',
                                            activation=tf.nn.relu)
        # Another 1D convolutional layer aftewards to generate more complex
        # time features. The kernel combines analyzes three consecutives
        # activation maps from the previous layer output.        
        conv2 = tf.keras.layers.Conv1D(filters=16,kernel_size=3,
                                            padding='same',
                                            activation=tf.nn.relu)
        # Performs global max pooling to "keep only the maximum" activation of
        # the previous convolutional layer filters. Technically answer where
        # (in time) the filter generated the strongest output.
        pooling1 = tf.keras.layers.GlobalMaxPooling1D()
        # Dense connecting layers to perform classification
        if self.use_regularizer:
            dense1 = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid,
                                                kernel_regularizer=
                                                tf.keras.regularizers.l2(0.1))    
        else:
            dense1 = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
        
        # Create connections and returns model
        inputs = tf.keras.Input((None,AirBinaryTemporalCNN.NUMBER_MFCC))
        x = conv1(inputs)
        x = conv2(x)
        x = pooling1(x)
        outputs = dense1(x)
        
        self.model = tf.keras.Model(inputs,outputs)
        
        return self.model
    
    '''
        Build efficient input pipelines for training and testing from tfrecord
        files
    '''
    def build_datasets(self,files,batch_size=128):
        
        # Dictionary containing features description for parsing purposes
       
        # Creates training data pipeline
        train_dataset = tf.data.TFRecordDataset([files[0]])
        train_dataset = train_dataset.map(AirBinaryTemporalCNN._parse_observation,
                                          num_parallel_calls=
                                          AirBinaryTemporalCNN.AUTOTUNE)
        train_dataset = train_dataset.map(AirBinaryTemporalCNN._transform_observation,
                                          num_parallel_calls=
                                          AirBinaryTemporalCNN.AUTOTUNE).batch(batch_size)
        # Creates test data pipeline
        test_dataset = tf.data.TFRecordDataset([files[1]])
        test_dataset = test_dataset.map(AirBinaryTemporalCNN._parse_observation,
                                        num_parallel_calls=
                                        AirBinaryTemporalCNN.AUTOTUNE)
        test_dataset = test_dataset.map(AirBinaryTemporalCNN._transform_observation,
                                        num_parallel_calls=
                                        AirBinaryTemporalCNN.AUTOTUNE).batch(batch_size)
        
        return train_dataset,test_dataset
    
    '''
        Parses observation from proto format
    '''
    def _parse_observation(example):
        observation = tf.io.parse_single_example(example, AirBinaryTemporalCNN.FEATURE_DESCRIPTION)
        observation['spec'] = tf.reshape(observation['spec'],(AirBinaryTemporalCNN.NUMBER_MFCC,-1))
        return observation
    
    '''
        Converts into correct format for training (input,output) = (spec,label)
    '''
    def _transform_observation(data):        
        return tf.transpose(data['spec']),data['label'] == b'Airbus'
    
    