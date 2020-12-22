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
  * Results with 2 classes: [here]()
