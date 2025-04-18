from __future__ import absolute_import
import os
import sys
import numpy as np
import tensorflow as tf
import scipy.misc 
try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x

from .osutils import mkdir_if_missing

from config import get_args
global_args = get_args(sys.argv[1:])

if global_args.run_on_remote:
  import moxing as mox
  mox.file.shift("os", "mox")

class Logger(object):
  def __init__(self, fpath=None):
    self.console = sys.stdout
    self.file = None
    if fpath is not None:
      if global_args.run_on_remote:
        dir_name = os.path.dirname(fpath)
        if not mox.file.exists(dir_name):
          mox.file.make_dirs(dir_name)
          print('=> making dir ', dir_name)
        self.file = mox.file.File(fpath, 'w')
        # self.file = open(fpath, 'w')
      else:
        mkdir_if_missing(os.path.dirname(fpath))
        self.file = open(fpath, 'w')

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    self.console.write(msg)
    if self.file is not None:
      self.file.write(msg)

  def flush(self):
    self.console.flush()
    if self.file is not None:
      self.file.flush()
      os.fsync(self.file.fileno())

  def close(self):
    self.console.close()
    if self.file is not None:
      self.file.close()


class TFLogger(object):
  def __init__(self, log_dir=None):
    """Create a summary writer logging to log_dir."""
    if log_dir is not None:
      mkdir_if_missing(log_dir)
    self.writer = tf.summary.create_file_writer(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""

    with self.writer.as_default():
      tf.summary.scalar(tag, value, step=step)
      self.writer.flush()

  def image_summary(self, tag, images, step):
    """Log a list of images."""

    with self.writer.as_default():
      # Create an Image object
      tf.summary.image(tag, images, step=step)
    
      self.writer.flush()
        
  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)
    

    with self.writer.as_default():
      # Create a histogram summary
      tf.summary.histogram(tag, hist, step=step)
      self.writer.flush() 


  def close(self):
    self.writer.close()