# air-noise
Here are some python scripts and jupyter notebooks using TF2.0 I have created to test new ideas related to the
automatic aircraft type recognition from the sound generated during take-off (doctorate thesis topic). This is not a complete repository
having everything that has been done and does not include the dataset either. I mainly use this material for teaching purposes now.

You can refer to some related papers here:
https://scholar.google.com/citations?user=q8RTRHkAAAAJ&hl=en

### Dataset

Description coming soon...

### Models

Some of the [models](https://github.com/alejandECE/air-noise/tree/master/python/models) tested are:
* A CNN with 1D Conv to extract overlapping windowed features (named air_temporal_cnn):
  * Results with 2 classes: [here](https://github.com/alejandECE/air-noise/blob/master/python/tests/air_two_classes_temporal_cnn_test.ipynb)
  * Results with 4 classes: [here](https://github.com/alejandECE/air-noise/blob/master/python/tests/air_four_classes_temporal_cnn_test.ipynb)
* A RNN where at each step (window) a CNN with 2D Conv is used to extract features:
  * Results with 2 classes: [here](https://github.com/alejandECE/air-noise/blob/beam-pipeline/python/tests/air_two_classes_rnn_test.ipynb)
  * Results with 4 classes: [here](https://github.com/alejandECE/air-noise/blob/master/python/tests/air_four_classes_rnn_test.ipynb)
  * Results with 8 classes: [here](https://github.com/alejandECE/air-noise/blob/beam-pipeline/python/tests/air_two_classes_rnn_test.ipynb)
* A [siamese network architecture](https://github.com/alejandECE/air-noise/blob/master/python/models/air_siamese_architecture.py) to be able to one-shot learned the encoding for any new observation. This is similar to the face recognition application where the sibling network is then used to find the encoding of any new face. Here the sibling network is used to find the encoding of any new aircraft. A [tf.data pipeline](https://github.com/alejandECE/air-noise/blob/bd565bae684e718324064e031579bdc8c00f4320/python/models/air_siamese_architecture.py#L102) is created to efficiently generate positive/negative pairs. Here are some visualizations of the embeddings found using PCA and Tensorboard:
  * [Embeddings notebook](https://github.com/alejandECE/air-noise/blob/master/python/tests/air_siamese_embeddings_visualization%201.ipynb)
  * Other visualizations (using Tensorboard):
![](/python/tests/air_siamese_tensorboard_visualization%201.png)
![](/python/tests/air_siamese_tensorboard_visualization%202.png)
![](/python/tests/air_siamese_tensorboard_visualization%203.png)
