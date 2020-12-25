# air-noise
Here are some python scripts and jupyter notebooks using TF2.0 I have created to test new ideas related to the
automatic aircraft type recognition from the sound generated during take-off (doctorate thesis topic). This is not a complete repository
having everything that has been done and does not include the dataset either. I mainly use this material for teaching purposes now.

You can refer to some related papers here:
https://scholar.google.com/citations?user=q8RTRHkAAAAJ&hl=en

### Dataset

The dataset consists of [188 sound measurements](python/extraction/db_exploration.ipynb) of aircrafts taking-off. Each measurement consists of 12 signals (sampled at 51.2kHz over 25 seconds) captured using a fully synchronized microphone array (sensors were placed distances ranging from 5 to 40cm apart). The array was used to help [localize the aircraft during take-off](https://www.sciencedirect.com/science/article/pii/S1051200414000979). As a result, we have 12*188 signals (with 25*51,200 samples each!). However, as you might expect all twelve signals from one measurement are highly correlated (actually this is why it can be localized!). These twelve signals although highly correlated contain suttle differences (each communication channel is not subjected to the same noise). We used all twelve signals (as a form of augmentation) if the measurement was in the training set, and only one selected at random if it was in the dev/test set. [Training/dev/test set splitting](python/extraction/tfrecord_dataset.py#L84) is done based on measurement not signal.

More details can be found in the scripts and notebooks [here](python/extraction).

### Models

Some of the [models](https://github.com/alejandECE/air-noise/tree/master/python/models) tested are:
* A [CNN](python/exports/2020-02-07%2001-09-35%20(four%20classes)/experiments/2020-07-02-13-50-21/diagrams/air_multiclass_temporal_cnn.jpg) with 1D Conv to extract overlapping windowed features (named air_temporal_cnn):
  * Results with 2 classes: [here](python/tests/air_two_classes_temporal_cnn_test.ipynb)
  * Results with 4 classes: [here](python/tests/air_four_classes_temporal_cnn_test.ipynb)
* A [RNN](python/exports/2020-03-01%2007-34-19%20(eight%20classes)/experiments/2020-07-02-14-24-54/diagrams/air_multiclass_rnn.jpg) where at each step (window) a CNN with 2D Conv is used to extract features:
  * Results with 2 classes: [here](python/tests/air_two_classes_rnn_test.ipynb)
  * Results with 4 classes: [here](python/tests/air_four_classes_rnn_test.ipynb)
  * Results with 8 classes: [here](python/tests/air_two_classes_rnn_test.ipynb)
* A [siamese network architecture](python/models/air_siamese_architecture.py) to be able to one-shot learned the encoding for any new observation. This is similar to the face recognition application where the sibling network is then used to find the encoding of any new face. Here the sibling network is used to find the encoding of any new aircraft. A [tf.data pipeline](python/models/air_siamese_architecture.py#L102) is created to efficiently generate positive/negative pairs. Here are some visualizations of the embeddings found using PCA and Tensorboard:
  * [Embeddings notebook](python/tests/air_siamese_embeddings_visualization%201.ipynb)
  * Other visualizations (using Tensorboard):
![](python/tests/air_siamese_tensorboard_visualization%201.png)
![](python/tests/air_siamese_tensorboard_visualization%202.png)
![](python/tests/air_siamese_tensorboard_visualization%203.png)
