"""Utility functions for videos, frame captures, plotting."""

import pandas as pd
from ast import literal_eval
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import config

# Returns paths for frame captures
def dataModules(root=None):
  """Loads a video from a file
    Args:
        root (str): The path to where the data is stored.
            Defaults to data directory specified in config file.
    Returns:
        Path to data root
    """

  if root is None:
    root = config.CONFIG.DATA_DIR

  return root

def scatterPlot(title="Plot", xlabel="", ylabel="", x1=[], y1=[], lineOfBestFit=True, save=False, 
                location="", plotName="Plot.png", alpha=0.5):
  fig = plt.figure()
  x = np.array(x1)
  y = np.array(y1)
  
  latexify()
  if lineOfBestFit:
    m, b = np.polyfit(x, y, 1)

    plt.plot(x, y, 'o', alpha=alpha)
    plt.plot(x, m*x + b)
    print("Line of Best Fit: " + str(str(m) + "x" + " + " + str(b)))
  else:
    plt.scatter(x, y, alpha=0.5)
  
  r_squared = calculatePlotData(x1, y1)
  print(r_squared)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()
  if save:
    fig.savefig(location + plotName)


def latexify():
  """Sets matplotlib params to appear more like LaTeX.
  Based on https://nipunbatra.github.io/blog/2014/latexify.html
  """
  params = {'backend': 'pdf',
            'axes.titlesize': 8,
            'axes.labelsize': 8,
            'font.size': 8,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'font.family': 'DejaVu Serif',
            'font.serif': 'Computer Modern',
            }
  plt.rcParams.update(params)

def calculatePlotData(x, y):
  """Calculated statistical data from calculations
    Args:
        x (list): list of x values
        y (list): list of y values
    Returns:
        The r-squared statistical value
    """

  correlation_matrix = np.corrcoef(x, y)
  correlation_xy = correlation_matrix[0,1]
  r_squared = correlation_xy**2
  
  return r_squared