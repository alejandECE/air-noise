#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import nptdms
import numpy as np
import os
import librosa
import datetime
from scipy.signal import decimate


class AircraftFeaturesExtractor(object):
  """
  Extracts specified features from TDMS files (grouped by some class)
  and exports features to a single file per signal per measurement
  """

  def __init__(self,
               dbcursor,
               classes,
               fs,
               feature_type='melspectrogram',
               segmentation=None,
               arrays=None,
               microphones=None,
               logged=True):
    self.fs = fs
    self.dbcursor = dbcursor
    self.classes = classes
    self.feature_type = feature_type
    self.segmentation = segmentation
    if arrays is None:
      self.arrays = [1, 2, 3]
    else:
      self.arrays = arrays
    if microphones is None:
      self.microphones = [1, 2, 3, 4]
    else:
      self.microphones = microphones
    self.logged = logged

  def build(self):
    """
    Goes through every class generating the corresponding npy files
    """
    # Creates a folder to store dataset
    self._create_root_folder()
    # Creates log file if chosen
    self._create_log()
    try:
      # Goes through every class finding measurements in DB
      for category in self.classes:
        # Logging
        self._log_message('Working on class: %s' % category)
        # Receives measurements for every aircraft in the class
        sql = '''SELECT m.id_measurement, m.url FROM 
                            measurements m WHERE m.aircraft IN ({:});'''.format(
          str(self.classes[category])[1:-1])
        self.dbcursor.execute(sql)
        results = self.dbcursor.fetchall()
        # Process all measurements from category
        self._process_measurements(results, category)
    except:
      # Logging
      self._close_log('An exception occurred...')
      return
    # Logging
    self._close_log('Done processing measurements!')

  def _process_measurements(self, measurements, category):
    """
    For every measurement in the input, reads every signal and extracts the corresponding features, storing the
    results in a npy file.
    """
    # Logging
    self._log_message('\tMeasurements included:')
    # Goes through every signal in the measurement
    path = self._create_category_folder(category)
    for measurement, url in measurements:
      # Logging
      self._log_message('\t\tId: %d, URL: %s' % (measurement, url))
      # Opens TDMS file
      tdms = nptdms.TdmsFile("../../datasets/classification/noise/" + url)
      for array in self.arrays:
        for microphone in self.microphones:
          # Reads data (actual time series)
          signal = tdms.channel_data(
            group='Untitled',
            channel='cDAQ1Mod' + str(array) +
                    '/ai' + str(microphone - 1))
          # Extracts features
          if self.segmentation is None:
            features = self._extract_features(signal)
          elif self.segmentation == 'tmid':
            window = 5
            tmid = self._get_tmid(measurement, array, microphone)
            start = int((tmid - window) * self.fs)
            end = int((tmid + window) * self.fs)
            features = self._extract_features(signal[start:end + 1])
          # Stores features
          name = 'm{}a{}s{}.npy'.format(measurement, array, microphone)
          np.save(os.path.join(path, name), features)

  def _extract_features(self, data):
    """
    Extracts the actual mel-spectrogram for the given data
    """
    if self.feature_type == 'melspectrogram':
      return self._extract_melspectrogram(data)

  def _get_tmid(self, measurement, array, microphone):
    """
    Gets the tmid point from the dataset for the given measurement, array and microphone
    """
    sql = ('''SELECT t.location FROM tmid t WHERE
               t.measurement = {:} AND
               t.array = {:} AND
               t.microphone = {:}''').format(measurement, array, microphone)
    self.dbcursor.execute(sql)
    result = self.dbcursor.fetchall()

    if len(result) > 0:
      return result[0][0]
    else:
      raise Exception('''No tmid information found in DB for 
                            measurement {}, array {} and microphone {}'''
                      .format(measurement, array, microphone))

  def _extract_melspectrogram(self, data):
    # Mel spectrogram configuration
    factor = 6
    mfcc = 128
    n_fft = 1024
    hop_length = 512
    # Decimate signal by factor (reducing sampling rate)
    data = decimate(data, factor)
    # Computes mel spectrogram
    spectrogram = librosa.feature.melspectrogram(np.asfortranarray(data),
                                                 sr=int(self.fs / factor),
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=mfcc)
    return spectrogram

  def _create_root_folder(self):
    self.root_path = 'exports/' + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir(self.root_path)

  def _create_category_folder(self, category):
    path = os.path.join(self.root_path, str(category))
    os.mkdir(path)
    return path

  def _create_log(self):
    if self.logged:
      self.log_file = open(os.path.join(self.root_path, 'log.txt'), 'w')

  def _log_message(self, msg):
    if self.logged:
      print(msg)
      self.log_file.write(msg + '\n')

  def _close_log(self, msg=None):
    if self.logged:
      if msg is not None:
        self._log_message(msg)
      self.log_file.close()


class AircraftDatasetFoldIterator:
  """
  Generates k-fold stratified folds for the aircraft dataset.

  It ensures the number of samples per class on each fold is as even as
  possible. During validation for traning it includes all signals from a
  measurement but for test only one random signal from each measurement
  is included.
  """

  def __init__(self, X, y, measurements, folds=5):
    self.current = 0
    self.X = X
    self.y = y
    self.measurements = measurements
    self.folds = folds
    self._splits = []
    for k in range(self.folds):
      self._splits.append([])

  def build(self):
    """
        Build k stratified folds using measurements and reponses provided
    """
    classes = np.unique(self.y)
    for category in classes:
      # Determines measurements for that class
      indexes = np.unique(self.measurements[self.y == category])
      indexes = np.random.permutation(indexes)
      # Distributes measurements the most evenly possible across folds
      overall_count = len(indexes)
      if overall_count < self.folds:
        raise Exception('''Class {} only has {} measurements, minimum
                                required is {} to create {} folds'''.format(
          category, overall_count, self.folds, self.folds))
      curr = 0
      fold_count = int(np.floor(overall_count / self.folds))
      for k in range(self.folds):
        for index in indexes[curr:curr + fold_count]:
          self._splits[k].append(index)
        curr += fold_count
      for index in indexes[curr:]:
        self._splits[np.random.randint(self.folds)].append(index)

  def __iter__(self):
    return self

  def __next__(self):
    # Builds test observations
    test_obs = []
    if self.current >= len(self._splits):
      self.current = 0
      raise StopIteration
    indexes = self._splits[self.current]
    for index in indexes:
      observations = (self.measurements == index).nonzero()[0]
      i = np.random.randint(len(observations))
      test_obs.append(observations[i])
    # Builds train observations
    train_obs = []
    for k, indexes in enumerate(self._splits):
      if k == self.current:
        continue
      for index in indexes:
        observations = (self.measurements == index).nonzero()[0]
        train_obs = train_obs + [i for i in observations]
        # Updates iteration
    self.current += 1
    # Return indexes
    return train_obs, test_obs
