#  Created by Luis A. Sanchez-Perez (alejand@umich.edu).
#  Copyright © Do not distribute or use without authorization from author

import pathlib
from collections import namedtuple
from typing import Tuple, List


# Evaluation comparison namedtuple
class Comparison(namedtuple('Comparison', ['distance', 'category'])):
  __slots__ = ()

  def __str__(self):
    return '({}: {:.2E})'.format(self.category, self.distance)


# Evaluation entry namedtuple
class EvaluationEntry(namedtuple('EvaluationEntry', ['time', 'category', 'comparisons'])):
  __slots__ = ()

  def __str__(self):
    return 'Entry ({:.2f} secs) - Expected: {}, Comparisons: {}'.format(
      self.time,
      self.category,
      ','.join([str(comparison) for comparison in self.comparisons])
    )


class EvaluationSummary(namedtuple('EvaluationSummary', ['accuracies'])):
  __slots__ = ()

  def __str__(self):
    output = '\nEvaluation Summary:\n'
    output += '\n'.join(['Accuracy (Top: {}): {:.2f}'.format(i + 1, acc) for i, acc in enumerate(self.accuracies)])
    return output


# Training log entry
class TrainingEntry(namedtuple('TrainingEntry', ['current', 'epochs', 'time', 'loss', 'metric'])):
  __slots__ = ()

  def __str__(self):
    return 'Epoch {} out of {} complete ({:.2f} secs) -- Batch Loss: {:.4f} -- Batch Acc: {:.2f}'.format(
      self.current,
      self.epochs,
      self.time,
      self.loss,
      self.metric
    )


class TrainingSummary(namedtuple('TrainingSummary', ['included', 'excluded'])):
  __slots__ = ()

  def __str__(self):
    output = '\nEvaluation Summary:'
    output += '\n\nIncluded categories:\n'
    output += '\n'.join([str(category) for category in self.included])
    output += '\n\nExcluded categories:\n'
    output += '\n'.join([str(category) for category in self.excluded])
    return output


# Opens the log file
def open_log(path: pathlib.Path):
  return open(path, 'w')


# Closes the log file
def close_log(log):
  log.close()


# Writes a step log to console
def write_step_log_to_console(entry) -> None:
  print(str(entry))


# Writes every step log to file
def write_step_log_to_file(file, entries: list) -> None:
  file.writelines([str(entry) + '\n' for entry in entries])


# Writes summary log to file
def write_summary_log_to_file(file, summary):
  file.write(str(summary))


# Writes summary log to console
def write_summary_log_to_console(summary):
  print(str(summary))
