#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import tensorflow as tf

# Path to saved model
model_path = None

# Most frequent eight classes used from dataset
AIRCRAFT_EIGHT_LABELS = [
  b'A320-2xx (CFM56-5)', b'B737-7xx (CF56-7B22-)', b'ERJ190 (CF34-10E)', b'B737-8xx (CF56-7B22+)',
  b'ERJ145 (AE3007)', b'A320-2xx (V25xx)', b'A319-1xx (V25xx)', b'ERJ170-175 (CF34-8E)'
]

# Most frequent four classes used from dataset
AIRCRAFT_FOUR_LABELS = [b'A320-2xx (CFM56-5)', b'B737-7xx (CF56-7B22-)', b'ERJ190 (CF34-10E)', b'B737-8xx (CF56-7B22+)']

# Two combined classes generated from dataset
AIRCRAFT_TWO_LABELS = [b'Airbus', b'Boeing']

# Autotune constant from tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
