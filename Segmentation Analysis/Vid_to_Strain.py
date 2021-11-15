#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from scipy import signal
import math
import typing
import pandas as pd
import scipy
import scipy.signal
from scipy.stats import linregress
from tqdm import tqdm
import re
import os, os.path
from os.path import splitext
import numpy as np
import shutil
from multiprocessing import dummy as multiprocessing
import time
from scipy import fftpack
import subprocess
import datetime
from datetime import date
import sys
import cv2
import matplotlib.pyplot as plt
import config
import loader
import sys
from shutil import copy
import math
import torch
import torchvision
import echonet
from shutil import copyfile
from numpy.linalg import norm
from scipy.signal import lfilter

def show(frame):
    cv2.imshow("test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_point(frame,x,y,color=(0,255,0),radius = 0):
    thickness = -1
    return cv2.circle(frame, (x,y), radius, color, thickness)

def plot_line(frame,p1,p2,color=(0,191,255),thickness = 1):
    return cv2.line(frame, (p1[0],p1[1]), (p2[0],p2[1]), color, thickness)

# Gets all the contours for certain image
def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count] = frame

    v = v.transpose((3, 0, 1, 2))

    return v
def savevideo(filename: str, array: np.ndarray, fps: typing.Union[float, int] = 1):
    """Saves a video to a file.

    Args:
        filename (str): filename of video
        array (np.ndarray): video of uint8's with shape (channels=3, frames, height, width)
        fps (float or int): frames per second

    Returns:
        None
    """

    c, f, height, width = array.shape

    if c != 3:
        raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(", ".join(map(str, array.shape))))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(f):
        out.write(array[:, i, :, :].transpose((1, 2, 0)))


def dist(x1,x2):
    return norm(x1-x2)

def select_contours(a):
    """
    This function selects the left ventricle from the set of contours
    It does this by selecting the contour with the mean position furthest from the bottom left hand corner
    """
    previous_index = 0
    previous_dist = 0
    for i in range(0,len(a)):
        if len(a[i])>50:
            mean_position = np.mean(np.mean(a[i],axis=-2),axis=0)
            distance = dist(mean_position,np.array([0,112]))
            if distance > previous_dist:
                previous_index = i
                previous_dist = distance
    return a[previous_index]

def obtain_lowest_points(img,thresh):
    """
    This function finds the endpoints of the contour by finding the lowest points of the minimum bounding box, this is a key function to improve
    the logic in this function is:
    a) find the contour
    b) find the smallest bounding rectangle
    c) find the appropriate set of indexes for the bottom two
    """

    # adjust the colors of the image for plotting
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # find the contours of the threshold
    im2, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # select the appropreate set of contours
    im2 = select_contours(im2)
    im2 = np.array(im2)

    # find the smallest bounding rectangle
    rec = cv2.minAreaRect(im2)
    color = (0, 0, 255) 
    thickness = 2
    box = cv2.boxPoints(rec)
    box = np.int0(box)

    # figure out which indexes are the bottom 2, using the largest y coordinate
    indexes = [0,1]
    for k in range(0,len(indexes)):
        for i in range(0,len(box)):
            if box[i][1]>box[indexes[k]][1] and not i in indexes:
                indexes[k]=i

    return img, [box[indexes[0]].tolist(),box[indexes[1]].tolist()]

# Filters
def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

def convolve_average(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

    re[0] = np.average(arr[:span])
    for i in range(1, span + 1):
        re[i] = np.average(arr[:i + span])
        re[-i] = np.average(arr[-i - span:])
    return re

def savgol(arr, span):  
    return scipy.signal.savgol_filter(arr, span * 2 + 1, 0)

def fft(arr, span):
    w = fftpack.rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return fftpack.irfft(w)

def passFilter(data, cutoff, fs, type="low"):
    b, a = signal.butter(cutoff, fs, type)
    y = signal.filtfilt(b, a, data)
    return y

#Peak/Valley Finding Algorithm
def method1(arr):
    x, y = [], []
        
    topNumPercent = int(len(arr) * 0.2)
    maxThreshold = sorted(arr, reverse=True)[:topNumPercent][-1]
    minThreshold = sorted(arr, reverse=False)[:topNumPercent][-1]
    
    for i in range(len(arr)):
        if arr[i] > maxThreshold:
            if arr[i] > arr[i - 1]:
                if arr[i] > arr[i + 1]:
                    x.append(arr[i])
        if arr[i] < minThreshold:
            if arr[i] < arr[i - 1]:
                if arr[i] < arr[i + 1]:
                    y.append(arr[i])
    return x, y, maxThreshold, minThreshold

#######
def calc_ratio(array,plot_dir,filename,vid=None,save = True,window_size = 3, smoothening_function="ConvolveAverage"):
    """
    This function calculates the ratio of the strain lengths, this is a key function to improve
    the lgoic in this function is:
    a) find the valleys using scipys find_peaks functions on the negative lengths
    b) find the peaks, using scipys find_peaks function on the lengths
    c) for each valley
        a) associate it with the nth peak
        b) if there is no nth peak
            a) find the maximum in the +/- 16 range
        take the ratio of the valley/peak
    return the ratios, excluding the first and last ratio
    """
    if save:
        plt.clf()
        plt.plot(array,label = 'Raw Data',alpha=0.7)
    array = [array[0]]+array+[array[-1]]

    if smoothening_function == "Moving Average":
        array = moving_average(array,window_size)
    elif smoothening_function == "ConvolveAverage":
        array = convolve_average(array, window_size)
    elif smoothening_function == "Savgol":
        array = savgol(array, window_size)
    elif smoothening_function == "FFT":
        array = fft(array, window_size)
    else:
        array = passFilter(array, window_size, 0.1, smoothening_function)
    

    if save:
        plt.plot(array,label = 'Moving Average: '+str(window_size),alpha=0.7)
    # Get peaks and valleys
    #x, y_array, maxThreshold, minThreshold = method1(array)
    # x = scipy.signal.find_peaks(-np.array(array),distance=32,width=5,prominence = 12)[0] # valley
    # y_array = scipy.signal.find_peaks(np.array(array),distance=32,width=5,prominence = 12)[0] # peak

    ratios = []
    maxVal, minVal = [], []
    maxIndex, minIndex = [], []

    # topNumPercent = int(len(array) * 0.2)
    # maxThreshold = sorted(array, reverse=True)[:topNumPercent][-1]
    # minThreshold = sorted(array, reverse=False)[:topNumPercent][-1]
    # plt.axhline(y=maxThreshold)
    # plt.axhline(y=minThreshold)
    # for i in range(len(array)):
    #     if array[i] > maxThreshold:
    #         if array[i] > array[i - 1]:
    #             if array[i] > array[i + 1]:
    #                 maxVal.append(array[i])
    #                 maxIndex.append(i)
    #     if array[i] < minThreshold:
    #         if array[i] < array[i - 1]:
    #             if array[i] < array[i + 1]:
    #                 minVal.append(array[i])
    #                 minIndex.append(i)
    
    # # Eliminating bad values from max and min points
    # for i in range(len(maxIndex)):
    #     if i < (len(maxIndex) - 1):
    #         if abs(maxIndex[i] - maxIndex[i + 1]) < 20:
    #             maxVal = maxVal[:i] + maxVal[i+1 :]
    
    # for i in range(len(minIndex)):
    #     if i < (len(minIndex) - 1):
    #         if abs(minIndex[i] - minIndex[i + 1]) < 20:
    #             minVal = minVal[:i] + minVal[i+1 :]

    # # Plotting the data
    # for i in range(len(maxIndex)):
    #     plt.plot(maxIndex[i], maxVal[i], 'ro')
    #     plt.plot(minIndex[i], maxVal[i], 'ro')

    # for i in range(len(maxVal)):
    #     ratios.append(minVal[i]/maxVal[i])

    # drop the first max (usually weird heartbeat). for each min, use the first max with the greater index.
    # if there are no remaining maxes, use the last max
    x = scipy.signal.find_peaks(-np.array(array),distance=32,width=5,prominence = 12)[0] # valley
    y_array = scipy.signal.find_peaks(np.array(array),distance=32,width=5,prominence = 12)[0] # peak

    ratios = []
    for i in range(0,len(x)): # for each valley
        if i < len(y_array): # if there is another peak
            x_val = array[x[i]]
            y_val = array[y_array[i]]
            if save:
                plt.scatter(x[i],x_val,color='green')
            if save:
                plt.scatter(y_array[i],y_val,color='red')
            ratios.append(x_val/y_val) # ratio is valley/peak

        else: # if there isnt another peak
            if save:
                plt.scatter(x[i],array[x[i]],color='green')
            if i==len(x)-1: # if this is the last valley, then the peak is the longest length in the remaining frames
                y = np.argmax(array[x[i]:])
                if save:
                    plt.scatter(y+x[i],array[y+x[i]],color='red')
                y_val = array[y+x[i]]
            else: # If this is not the last valley, then the peak is the longest length within 16 frames of the valley
                delta = min([x[i],16])
                y = np.argmax(array[max([x[i]-16,0]):min([x[i]+16,len(array)])])
                if save:
                    plt.scatter(y+x[i]-delta,array[y+x[i]-delta],color='red')
                y_val = array[y+x[i]-delta]
            x_val = array[x[i]]
            
            ratios.append(x_val/y_val)
    if save:
        plt.savefig(os.path.join(plot_dir,filename[:-3]+'png'))
        plt.clf()
    ratios.sort()
    return ratios

def midpoint(point1,point2):
    return (point1+point2)/2

def smooth(points):
    """
    Smooth a set of points by finding the midpoint between every pair
    """
    point_arr = [points[0]]
    for i in points[1:]:
        point_arr.append(midpoint(point_arr[-1],i))
    return np.array(point_arr)

def get_points(vid,thresh):
    """
    This function finds the left and rightmost bottom points for the segmentation
    this would be a key area to improve, as the logic of separating this is not fully fleshed out
    
    Current function logic:
    for each frame
        find a bounding box of the contour
        order the bottom two points by left and right
    take a moving average of the points (half the distance each point moves between frames)
    return the left and right points
    """
    
    first = np.transpose(vid,(1,2,3,0))
    video = []
    guess = None
    vertexes = []
    for frame in range(0,len(first)):
        # obtain_lowest_points returns the bottom two points of the smallest bounding box
        img,points = obtain_lowest_points(first[frame],thresh[frame].copy())
        if np.linalg.norm(points[0])>np.linalg.norm(points[1]):
            c = points[0].copy()
            points[0] = points[1].copy()
            points[1] = c
        vertexes.append(points)

    ok = np.array(vertexes)
    left_points = smooth(ok[:,0,:])
    right_points = smooth(ok[:,1,:])
    return left_points,right_points

def get_dilation(threshes,dilations = 1):
    """
    This function dilates the segmentation repeatedly, to improve smoothing
    """
    dilated = []
    for t in threshes:
        cv2_object = t.copy()
        cv2_object = cv2.dilate(cv2_object, None, iterations=dilations)
        
        dilated.append(cv2_object)
    
    return np.array(dilated)

def distance_calc(x1,x2):
    """
    This function calculates the total length of the contour, accounting for the connections it has to make.
    """
    total_length = 0
    for i in range(0,len(x1)-1):
        total_length += dist(x1[i,0],x1[i+1,0])
    total_length += dist(x1[-1,0],x2[0,0])
    for i in range(0,len(x2)-1):
        total_length += dist(x2[i,0],x2[i+1,0])
    return total_length

def strain_lengths(vid,threshes,first_points,second_points,filename,strain_dir,plot_dir,excel_dir, window_size = 3,downsample = 2,contour_thickness = 1, point_radius = 0):
    """
    Function for estimating the strain length. It takes in as an input:
    vid: The video, for producing graphics
    threshes: The segmentation, as a binary numpy array
    first_points: The left point of the contour
    second_points: the right point of the contour
    filename: the output filename, for saving graphics
    strain_dir, plot_dir, excel_dir: the locations to save produced graphics
    window_size: The size of the window for producing a moving average
    downsample: the downsample rate for dealing with the coastline problem
    contour_thickness: the contour thickness, for graphics production
    point_radius: the plotted point thickness, for graphics production
    
    The logic behind the code is as follows:
    a) transpose the video, make a spare, and produce an array for each of the variables that will be saved in the excel document
    b) loop over each frame
    c) Find the contours of the segmentation
        a) select the optimal contour, the one most likely to be the left ventricle
        b) Find the points on the contour closest to the left and right points that were inputed
        c) split the contour into a left and right half, and downsample it to deal with the coastline problem
        d) Plot and save information for graphical and csv purposes
        e) calculate the length of the contour on the frame
    d) calculate the strain from the length in each frame
    """
    # spare videos
    first = np.transpose(vid,(1,2,3,0))
    spare = first.copy()
    
    thresh = threshes
    
    # CSV arrays
    frame_num = []
    x1s = []
    y1s = []
    error1 = []
    x2s = []
    y2s = []
    error2 = []
    length = []
    angle = []
    
    # loop over each frame
    for frame in range(0,len(first)):
        
        # adjust the color
        rgb = cv2.cvtColor(first[frame,:,:], cv2.COLOR_BGR2RGB)
        
        # Grab the specific threshold we are investigating, convert multiply it by 255 (as its a binary array, this will make it either 0 or 255), then rebinarize it based on being equal to 255
        t = (thresh[frame,:,:]*255).copy()
        t = cv2.inRange(t, 250, 255)
        
        # Find the contours of the threshold, and select the left ventricle
        im2, contours = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = select_contours(im2)
        
        # find the left end of the contour (x1,y1), and the right end of the contour (x2,y2)
        x1 = int(first_points[frame,0])
        y1 = int(first_points[frame,1])
        x2 = int(second_points[frame,0])
        y2 = int(second_points[frame,1])
        
        # calculate the angle between the two points for csv
        def calc_degrees(point):
            return math.atan2(point[1], point[0])
        deg = calc_degrees([x1-x2,y1-y2])
        angle.append(deg)
        
        # starting from the first point in the contour, find the two points closest two the left and right ends of the contour
        final_point = im2[0][0]
        index = 0
        count = 0
        for j in im2:
            count+=1
            if np.linalg.norm(j[0]-np.array([x1,y1]))< np.linalg.norm(final_point-np.array([x1,y1])):
                final_point = j[0]
                index = count
        
        final_point2 = im2[0][0]
        index2 = 0
        count2 = 0
        for j in im2:
            count2+=1
            if np.linalg.norm(j[0]-np.array([x2,y2]))< np.linalg.norm(final_point2-np.array([x2,y2])):
                final_point2 = j[0]
                index2 = count2
        
        # Split the contour into a left and right halfs, and then downsample it
        left_contour = im2[:index].copy()
        left_contour = left_contour[::downsample]
        right_contour = im2[index2:].copy()
        right_contour = right_contour[::downsample]
        
        bottom_contour = im2[index:index2].copy()
        bottom_contour = bottom_contour[::downsample]
        # plot the contour for graphics production
        for k in range(0,len(left_contour)-1):
            plot_line(first[frame],left_contour[k][0],left_contour[k+1][0],color = (0,255,0))
            
        # the last element of the right contour connects to the first element of the right
        plot_line(first[frame],left_contour[0][0],right_contour[-1][0],color = (0,255,0))
        for k in range(0,len(right_contour)-1):
            plot_line(first[frame],right_contour[k][0],right_contour[k+1][0],color = (0,255,0))
        
        for k in range(0,len(bottom_contour)-1):
            plot_line(first[frame],bottom_contour[k][0],bottom_contour[k+1][0],color = (0,0,255))
        
        # graphically connect the contours to the left and right endpoints
        plot_line(first[frame],right_contour[0][0],[x2,y2],color = (0,255,0),thickness = contour_thickness) # im2[index2-1][0]
        plot_line(first[frame],left_contour[-1][0],[x1,y1],color = (0,255,0),thickness = contour_thickness) # im2[index][0]
        plot_point(first[frame],x1,y1,color=(255,0,255),radius=point_radius)
        plot_point(first[frame],x2,y2,color=(255,0,255),radius=point_radius)
        
        # append calculations and information for graphical purposes
        frame_num.append(frame)
        x1s.append(final_point[0])
        y1s.append(final_point[1])
        error1.append(np.linalg.norm(final_point-np.array([x1,y1]))) # the distance between the last point on the contour and its associated left point
        x2s.append(final_point2[0])
        y2s.append(final_point2[1])
        error2.append(np.linalg.norm(final_point2-np.array([x2,y2])))
        
        # calculate the total length
        total_length = distance_calc(right_contour,left_contour)+dist(right_contour[0][0],[x2,y2])+dist(left_contour[-1][0],[x1,y1])
        
        length.append(total_length)
        
    
    video = np.transpose(np.array(first),(3,0,1,2))

    savevideo(os.path.join(strain_dir,filename),video,fps=30)
    final = pd.DataFrame({'frame_num':frame_num,'x1':x1s,'y1':y1s,'error_1':error1,'x2':x2s,'y2':y2s,'error_2':error2,'length':length,'angle':angle})
    final.to_csv(os.path.join(excel_dir,filename[:-4]+'.csv'))
    return final,calc_ratio(length,plot_dir,filename,window_size = window_size)


def estimate_strain(input_vid,weights,segmentation_dir,strain_dir,plot_dir,excel_dir,dilations = 1,segmenter = None,flip=False,window_size=3, downsample = 2,output_filename = None,contour_thickness = 1, point_radius = 0):
    """ 
    Single Function to estimate the strain for any input video, this function should, in additional to calculating the strain, provide the option for saving and producing a plot of contour length by frame, a csv of contour length by frame, and a video of the contour
    The current logic for this function is:
    a) Segment the video
    b) Load the segmentation in as a numpy array
    c) Find the bottom left and right points of the segmentation
    d) Dilate the segmentation
    e) estimate the strain length
    """
    if output_filename is None:
        output_filename = os.path.basename(input_vid)
        
    if segmenter is None:
        segmenter = Segmentation(weights)
    
    # produce the segmentation
    video_file = segmenter.single_vid_prediction(input_vid,segmentation_dir,flip=flip)
    loaded_vid = loadvideo(video_file)
    thresh = (np.load(video_file[:-4]+'.npy')>0).astype(np.uint8) #video_file[:-4]+
    
    # find left and right bottom points
    left,right = get_points(loaded_vid,thresh)
    
    # dilate the segmentation
    thresh = get_dilation(thresh,dilations = dilations)
    
    # estimate the strain
    measure = strain_lengths(loaded_vid,thresh,left,right,output_filename,strain_dir,plot_dir,excel_dir,window_size=window_size,downsample = downsample,contour_thickness = contour_thickness, point_radius = point_radius)
    return measure[1]

def segment(inp):
  """Uses pre-trained weights from EchoNet-Dynamic to
    create segmentation of left ventricular region
  Args:
    inp (arr): array object of image to segment
  Returns:
    None
  """

  x = inp.transpose([2, 0, 1])  #  channels-first
  x = np.expand_dims(x, axis=0)  # adding a batch dimension    
    
  mean = x.mean(axis=(0, 2, 3))
  std = x.std(axis=(0, 2, 3))
  x = x - mean.reshape(1, 3, 1, 1)
  x = x / std.reshape(1, 3, 1, 1)

  with torch.no_grad():
    x = torch.from_numpy(x).type('torch.FloatTensor').to(device)
    output = model(x)    

  y = output['out'].numpy()
  y = y.squeeze()

  out = y>0

  mask = inp.copy()
  mask[out] = np.array([0, 0, 255])

def smooth_segmentation(x,alpha):
    """
    This function smooths the segmentation over between frames.
    The logic is:
    the nth frame = alpha * the nth frame + (1-alpha) * the n-1th frame
    """
    final = [x[0]]
    for i in x[1:]:
        final.append(alpha*i+(1-alpha)*final[-1])
    return np.array(final)

def collate_fn(x):
    x, f = zip(*x)
    i = list(map(lambda t: t.shape[1], x))
    x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
    return x, f, i

class Segmentation:
    """
    This class exists to avoid having to load the segmentation model for each video.
    """
    def __init__(self, segmentation_model_checkpoint, mean = np.array([31.834011, 31.95879,  32.082172]) , std = np.array([48.866325, 49.137333, 49.361984])):
        self.mean = mean
        self.std = std
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, aux_loss = False)
        self.model.classifier[-1] = torch.nn.Conv2d(self.model.classifier[-1].in_channels, 1, kernel_size = self.model.classifier[-1].kernel_size)
        
        self.model = torch.nn.DataParallel(self.model)
        checkpoint = torch.load(segmentation_model_checkpoint, map_location="cpu")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def single_vid_prediction(self,vid,output_folder,flip=True,alpha = 0.9):
        """
        This function exists to segment a video. the alpha value smooths the segmentation between frames, adjusting its value would increase its smoothing
        """
        device = torch.device("cpu")
        kwargs = {"target_type": "Filename",
                  "mean": self.mean,
                  "std": self.std
          }
        block = 1024
        try:
            shutil.rmtree('temp')
            os.mkdir('temp')
        except:
            os.mkdir('temp')
        try:
            os.mkdir(output_folder)
        except:
            x=1
        copyfile(vid, "temp/" + vid.split('/')[-1])
        test_ds = echonet.datasets.Echo(split = "external_test", external_test_location = 'temp', length = None, period = 1, **kwargs)
        test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size = 10, num_workers = 0, shuffle = False, pin_memory=(device.type == "cuda"), collate_fn = collate_fn)
        with torch.no_grad():
            for (x,f,i) in test_dataloader:
                if flip:
                    x = torch.flip(x,[3])
                x = x.to(device)
                y = np.concatenate([self.model(x[i:(i+block),:,:,:])["out"].detach().cpu().numpy() for i in range(0, x.shape[0], block)]).astype(np.float16)
                start = 0
                oldvideo = x.cpu().numpy().copy()
                oldvideo = oldvideo * self.std.reshape(1, 3, 1, 1)
                oldvideo = oldvideo + self.mean.reshape(1, 3, 1, 1)
                newvideo = oldvideo.copy()
                # This is the line that smooths the segmentation
                newvideo[:,2,:,:] = np.maximum(newvideo[:,2,:,:], 255. * (smooth_segmentation(y[:, 0, :, :],alpha) > 0))
                for (filename, offset) in zip(f,i):
                    # This line also smooths the segmentation
                    np.save(os.path.join(output_folder, os.path.splitext(filename)[0]), smooth_segmentation(y[start:(start+offset), 0, :, :],alpha))
                    #plain videos
                    echonet.utils.savevideo(os.path.join(output_folder,os.path.splitext(filename)[0] + ".avi"), np.transpose(newvideo[start:(start+offset), :, :, :],(1,0,2,3)).astype(np.uint8), 50)
                    shutil.rmtree('temp')
                    return os.path.join(output_folder,os.path.splitext(filename)[0] + ".avi")

# root = loader.dataModules()
# print(root)
# weights_path = os.path.join(root, "Weights-20201103T193519Z-001", "Weights", "deeplabv3_resnet50_random.pt")
# segmentations_path = os.path.join(root, "Ishan")
# strains_path = os.path.join(root, "Ishan", )
# plot_path = os.path.join(root, "Ishan")
# excel_path = os.path.join(root, "Ishan")

# estimate_strain("/Users/ishan/Box/Strain Study/StrainStudyA4c-Preprocessed/2PXGZZJY_1_EPIQ7C_NO.avi", flip=False, weights=weights_path, segmentation_dir=segmentations_path,
#                         strain_dir=strains_path, plot_dir=plot_path, excel_dir=excel_path, dilations=1, point_radius=1)