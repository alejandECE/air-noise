#  Created by Luis Alejandro (alejand@umich.edu)
from nptdms import TdmsFile
import numpy as np
import pandas as pd
import os
import librosa
import datetime
from scipy.signal import decimate

class AircraftDatasetBuilder(object):
    '''
        Builds a aircraft dataset using the specified feature extraction method
    '''
    def __init__(self,
                 dbcursor,
                 classes,
                 fs,
                 feature_type = 'ngram_mfcc',
                 train_pct = 0.7,
                 segmentation = 'db_original',
                 arrays = [1,2,3],
                 microphones = [1,2,3,4],
                 verbose = True):
        
        self.fs = fs
        self.dbcursor = dbcursor
        self.classes = classes
        self.feature_type = feature_type
        self.train_pct = train_pct
        self.segmentation = segmentation
        self.arrays = arrays
        self.microphones = microphones
        self.verbose = verbose
        self._init_sets()
        
    def _init_sets(self):
        self.sets = dict()
        if self.segmentation == 'db_original':
            for i in range(4):
                self.sets[i] = dict()
                self.sets[i]['X'] = list()
                self.sets[i]['y'] = list()
                self.sets[i]['extra'] = list()
        elif self.segmentation == 'db_largest_merge':
            self.sets[0] = dict()
            self.sets[0]['X'] = list()
            self.sets[0]['y'] = list()
            self.sets[0]['extra'] = list()
                
    def build(self):
        for category in self.classes:
            if self.verbose == True:
                print('Working on class: %s' % category)
            # Receives measurements for every aircraft in the class
            sql = '''SELECT m.id_measurement, m.url FROM 
                        measurements m WHERE m.aircraft IN ({:});'''.format(
                        str(self.classes[category])[1:-1])
            self.dbcursor.execute(sql)
            results = self.dbcursor.fetchall()
            self._update_sets(results, category)
        return self.sets

    def _update_sets(self, measurements, category):
        if self.verbose == True:
            print('\tMeasurements included:')
        for measurement, url in measurements:
            if self.verbose == True:
                print('\t\tId: %d, URL: %s' % (measurement,url))
            tdms = TdmsFile("../../datasets/classification/noise/" + url)
            for array in self.arrays:
                for microphone in self.microphones:
                    signal = tdms.channel_data(
                            group = 'Untitled',
                            channel = 'cDAQ1Mod' + str(array) +
                            '/ai' + str(microphone - 1))
                    segments = self._extract_segments(measurement,microphone,array)
                    for i, (start, end) in enumerate(segments):
                        features = self._extract_features(signal[start:end+1])
                        features = features.flatten()
                        self.sets[i]['X'].append(features)
                        self.sets[i]['y'].append(category)
                        self.sets[i]['extra'].append(measurement)
       
    def _extract_segments(self, measurement,microphone,array):
        if self.segmentation == 'db_original':
            return self._load_db_original_segments(measurement,microphone,array)
        elif self.segmentation == 'db_largest_merge':
            return self._load_db_largest_merge_segments(measurement,microphone,array)
    
    def _extract_features(self, data):
        if (self.feature_type == 'ngram_mfcc'):
            return self._extract_ngram_mfcc(data)
        
    def _load_db_original_segments(self,measurement,microphone,array):
        segments = list()
        sql = ('''SELECT s.start, s.end FROM segments s WHERE
               s.measurement6 = {:} AND
               s.array = {:} AND
               s.microphone = {:} AND
               type = 2''').format(measurement, array, microphone)
        self.dbcursor.execute(sql)
        result = self.dbcursor.fetchall()
                        
        for (start,_) in result:
            start = int(start * self.fs)
            end = int(start + 2*self.fs) # 2 secs exactly
            segments.append((start,end))
        return segments
              
    def _load_db_largest_merge_segments(self,measurement,microphone,array):
        segments = list()
        sql = ('''SELECT t.location FROM tmid t WHERE
               t.measurement = {:} AND
               t.array = {:} AND
               t.microphone = {:}''').format(measurement, array, microphone)
        self.dbcursor.execute(sql)
        result = self.dbcursor.fetchall()
        
        if len(result) > 0:                
            tmid = result[0][0]
            start = int((tmid - 4)*self.fs) 
            end = int((tmid + 4)*self.fs)
            segments.append((start,end))
            
        return segments
      
    def _extract_ngram_mfcc(self, data, size=(20,), d_factor=6):
        mfcc_size = size[0]
        data = decimate(data,d_factor)
        spectrogram = librosa.feature.mfcc(data,int(self.fs / d_factor),n_mfcc=mfcc_size,hop_length=512)
        if len(size) > 1:
            t_size = size[1]
            if t_size > spectrogram.shape[1]:
                spectrogram = np.pad(spectrogram,((0,0),(0,t_size - spectrogram.shape[1])),mode='edge')
            else:
                spectrogram[:,:t_size]
        else:
            return spectrogram
                   
    def export_to_csv(self):
        path = 'exports/' + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        os.mkdir(path)
    
        for i in self.sets:
            X = pd.DataFrame(np.array(self.sets[i]['X']), columns=['feature %d' % feature for feature in range(len(self.sets[i]['X'][0]))])
            y = pd.DataFrame(np.array(self.sets[i]['y']), columns=['class'])
            extra = pd.DataFrame(self.sets[i]['extra'], columns=['measurement'])
            dataset = X.join(y).join(extra)
            dataset.to_csv(path + '/segment_%d.csv' % i, index=False) 
                    
def aircraft_dataset_split(predictors, responses, measurements, holdout_pct=0.3, return_measurements=False):
    '''
        Splits the aircraft dataset into training and hold out sets.
        It ensures the percentage of holdout samples of each class is the same.
        For the traning set it includes all signals from a measurement but for
        the holdout set only one random signal from each measurement is
        included.
    '''    
    # Shuffles dataset
    m = len(measurements)
    i = np.random.permutation(range(m))
    predictors = predictors[i,:]
    responses = responses[i]
    measurements = measurements[i]
    # Splits dataset
    train_obs = np.zeros(m,dtype=bool)
    holdout_obs = np.zeros(m,dtype=bool)
    classes = np.unique(responses)
    for category in classes:
        indexes = np.unique(measurements[responses == category])
        indexes = np.random.permutation(indexes)
        k = int(len(indexes) * (1 - holdout_pct))
        # Training set
        for index in indexes[:k]:
            new_obs = (measurements == index)
            train_obs = train_obs | new_obs
        # Holdout set
        for index in indexes[k:]:
            new_obs = (measurements == index)
            mask = np.random.permutation(new_obs.nonzero()[0])
            new_obs[mask[:-1]] = False
            holdout_obs = holdout_obs | new_obs
    X = predictors[train_obs,:]           
    y = responses[train_obs]   
    X_holdout = predictors[holdout_obs,:]           
    y_holdout = responses[holdout_obs]
    takeoffs = measurements[train_obs]
    takeoffs_holdout = measurements[holdout_obs]

    if return_measurements:
        return X,y,X_holdout,y_holdout,takeoffs,takeoffs_holdout
    else:
        return X,y,X_holdout,y_holdout

class AircraftDatasetFoldIterator():
    '''
        Generates k-fold stratified folds for the aircraft dataset.
        
        It ensures the number samples per class on each fold is as even as
        possible. During validation for traning it includes all signals from a
        measurement but for test only one random signal from each measurement
        is included.
    '''
    def __init__(self,X,y,measurements,folds=5):
        self.current = 0
        self.X = X
        self.y = y
        self.measurements = measurements
        self.folds = folds
        self._splits = []
        for k in range(self.folds):
            self._splits.append([])

    def build(self):
        '''Build k stratified folds using measurements and reponses provided'''
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
                                category,overall_count,self.folds,self.folds))
            curr = 0
            fold_count = int(np.floor(overall_count / self.folds))
            for k in range(self.folds):
                for index in indexes[curr:curr+fold_count]:
                    self._splits[k].append(index)
                curr += fold_count
            for index in indexes[curr:]:
                self._splits[np.random.randint(self.folds)].append(index)

    def __iter__(self):
        return self

    def __next__(self):
        # Builds test observations
        test_obs = []
        if (self.current >= len(self._splits)):
            raise StopIteration
        indexes = self._splits[self.current]
        for index in indexes:
            observations = (self.measurements == index).nonzero()[0]
            i = np.random.randint(len(observations))
            test_obs.append(observations[i])
        # Builds train observations
        train_obs = []
        for k,indexes in enumerate(self._splits):
            if k == self.current:
                continue
            for index in indexes:
                observations = (self.measurements == index).nonzero()[0]
                train_obs = train_obs + [i for i in observations]                
        # Updates iteration
        self.current += 1
        # Return indexes
        return (train_obs,test_obs)