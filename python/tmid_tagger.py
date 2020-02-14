#  Created by Luis Alejandro (alejand@umich.edu)
import mysql.connector
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import numpy as np
from scipy.signal import spectrogram
from scipy.signal import decimate
import librosa
from tmid_tagger_utils import SoundPlayer, SoundHead
from multiprocessing import Pipe


class AircraftTmidTagger(object):
  ALPHA = 1.

  def __init__(self):
    self.fs = 51200
    self.factor = 6
    # Connects to DB
    self._mydb = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="cuba",
      database="airnoise"
    )
    # Gets signals
    dbcursor = self._mydb.cursor()
    sql = '''SELECT t.measurement, t.array, t.microphone, t.location FROM
                tmid t ORDER BY t.measurement, t.array, t.microphone;'''
    dbcursor.execute(sql)
    self._signals = dbcursor.fetchall()
    # Auxiliary pointers to iterate through signals
    self._measurement = None
    self._current = 0
    # Info about user selected tmid
    self._tmid = None
    self._tmid_history = []
    # Creates figure
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(131)  # signal
    fig.add_subplot(132)  # spectrogram
    fig.add_subplot(133)  # mfcc
    fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    fig.canvas.mpl_connect('button_release_event', self._on_click)
    fig.canvas.mpl_connect('key_release_event', self._on_key_release)
    fig.tight_layout(w_pad=4, rect=[0.03, 0.03, 0.97, 0.95])
    fig.show()
    self.fig = fig
    self._is_shift_pressed = False
    # Sound and graphic helpers
    self._sound = None
    self._head = None
    # Plots first signal
    self._plot_curr_signal()
    # Start figures updater
    self._timer = self.fig.canvas.new_timer(interval=100)
    self._timer.add_callback(self._update_gui)
    self._timer.start()
    plt.show()

  def _update_gui(self):
    self.fig.canvas.draw()

  def _on_key_release(self, event):
    if event.key == 'shift':
      print('Release')
      self._is_shift_pressed = False

  def _on_key_press(self, event):
    print(event.key)
    if event.key == ' ':
      self._play_curr_signal()
    elif event.key == 'right' or event.key == 'shift+right':
      if self._is_shift_pressed:
        self._change_signal(step=12)
      else:
        self._change_signal(step=1)
    elif event.key == 'left' or event.key == 'shift+left':
      if self._is_shift_pressed:
        self._change_signal(step=-12)
      else:
        self._change_signal(step=-1)
    elif event.key == 'a':
      if self._sound is not None and self._sound.is_alive():
        self._sound.rewind()
    elif event.key == 'd':
      if self._sound is not None and self._sound.is_alive():
        self._sound.forward()
    elif event.key == 'shift':
      self._is_shift_pressed = True

  def _on_click(self, event):
    if event.inaxes == self.fig.get_axes()[0]:
      location = event.xdata
      self._update_user_tmid(self.fig.get_axes()[0], location)

  def _change_signal(self, step=1):
    # stops playing current signal
    self._stop_curr_signal()
    # moves pointer to next signal in the list
    self._current += step
    if self._current >= len(self._signals):
      self._current = self._current - len(self._signals)
    if self._current < 0:
      self._current = len(self._signals) + self._current
    # plot current signal
    self._plot_curr_signal()
    # resets history of tmids
    self._tmid_history.clear()

  def _play_curr_signal(self):
    self._stop_curr_signal()
    data, fs, _ = self._load_curr_signal()
    conn_head, conn_player = Pipe()
    self._head = SoundHead(self.fig.get_axes()[0], conn_head)
    self._head.start()
    self._sound = SoundPlayer(data, fs, conn_player)
    self._sound.start()

  def _stop_curr_signal(self):
    if self._sound is not None and self._sound.is_alive():
      self._sound.stop()
    if self._head is not None and self._head.is_alive():
      self._head.stop()

  # Loads current measurement info from TDMS file
  def _load_measurement(self, measurement):
    dbcursor = self._mydb.cursor()
    sql = '''SELECT m.url FROM measurements m WHERE
                    m.id_measurement = {:}'''.format(measurement)
    dbcursor.execute(sql)
    results = dbcursor.fetchall()
    url = results[0][0]
    self.tdms = TdmsFile("../../datasets/classification/noise/" + url)
    self._measurement = measurement

  # Loads time series data for current signal (measurement, array, microphone)
  def _load_curr_signal(self):
    # Retrieve info from current signal
    info = self._signals[self._current]
    measurement, array, microphone, _ = info
    # Opens measurement if is not open yet
    if measurement != self._measurement:
      self._load_measurement(measurement)
    # Loads and prepares data
    data = self.tdms.channel_data(group='Untitled', channel='cDAQ1Mod'
                                                            + str(array) + '/ai' + str(microphone - 1))
    data = data / max(np.absolute(data))
    factor = self.factor
    data = decimate(data, factor)
    fs = int(self.fs / factor)

    return data, fs, info

  # Plots data and processing results for current signal
  def _plot_curr_signal(self):
    # Loads data
    data, fs, info = self._load_curr_signal()
    measurement, array, microphone, location = info
    # Retrieves axes
    axes = self.fig.get_axes()
    # Plots signal
    self._plot_timeseries(axes[0], data, fs)
    # Plots spectrogram
    self._plot_spectrogram(axes[1], data, fs)
    # Plots mfcc
    self._plot_mfcc(axes[2], data, fs)
    # Plots tmids
    self._plot_user_tmid(axes[0], location)
    self._plot_time_tmid(axes[0], data, fs)
    self._plot_freq_tmid(axes[0], data, fs)
    # Updates title
    self.fig.suptitle('Measument: {}, Arr: {}, Mic: {}'.format(measurement,
                                                               array, microphone), fontsize=20, fontweight='bold')

  def _plot_time_tmid(self, ax, data, fs):
    samples = len(data)
    overlap = 0.2
    size = int(fs / 2)
    indexes = list(range(0, samples, int(size * (1 - overlap))))
    if indexes[-1] + size > samples:
      del indexes[-1]
    indexes = np.array(indexes)
    energy = np.zeros(indexes.shape)
    for i, location in enumerate(indexes):
      energy[i] = (data[location:location + size] ** 2).mean()
    i = energy.argmax()
    if i > len(indexes) - 2:
      i = len(indexes) - 2
    location = int((indexes[i + 1] + indexes[i]) / 2) / fs
    print(location)
    ax.plot([location, location], [-1, 1], 'm-', alpha=self.ALPHA, lw=2)

  def _plot_freq_tmid(self, ax, data, fs):
    f, t, Sxx = spectrogram(data, fs, mode='magnitude',
                            window='hamming', nperseg=512, noverlap=256)
    power = Sxx.mean(axis=0)
    location = t[power.argmax()]
    ax.plot([location, location], [-1, 1], 'c-', alpha=self.ALPHA, lw=2)

  def _plot_user_tmid(self, ax, location):
    self._tmid, = ax.plot([location, location], [-1, 1], 'k-', alpha=self.ALPHA, lw=2)

  def _update_user_tmid(self, ax, location):
    print(location)
    self._tmid.set_data([location, location], [-1, 1])
    info = self._signals[self._current]
    measurement, array, microphone, _ = info
    dbcursor = self._mydb.cursor()
    sql = '''UPDATE tmid t SET t.location = {} WHERE t.measurement = {}
            AND t.array = {} AND t.microphone = {}'''.format(
      location, measurement, array, microphone)
    dbcursor.execute(sql)
    self._mydb.commit()
    self._signals[self._current] = (measurement, array, microphone, location)

  # Plots time series in the given axes
  def _plot_timeseries(self, ax, data, fs):
    t = np.arange(len(data)) / fs
    ax.clear()
    ax.plot(t, data)
    ax.grid(True)
    self._set_axis_font(ax)

  # Plots spectrogram in the given axes
  def _plot_spectrogram(self, ax, data, fs):
    ax.clear()
    f, t, Sxx = spectrogram(data, fs, mode='magnitude',
                            window='hamming', nperseg=512, noverlap=256)
    # Sxx = 20*np.log(Sxx)
    ax.pcolormesh(t, f, Sxx, cmap='magma')
    self._set_axis_font(ax)
    print(Sxx.max())
    print(Sxx.min())

  # Plots MFCC in the given axes
  def _plot_mfcc(self, ax, data, fs):
    mat = librosa.feature.mfcc(np.asfortranarray(data), fs, n_mfcc=100, hop_length=512, fmax=1000)
    ax.clear()
    ax.pcolormesh(mat, cmap='magma')
    self._set_axis_font(ax)
    print(mat.shape)

  # Sets axis font for the given axes
  def _set_axis_font(self, ax, fontsize=18):
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(fontsize)


if __name__ == "__main__":
  tagger = AircraftTmidTagger()
