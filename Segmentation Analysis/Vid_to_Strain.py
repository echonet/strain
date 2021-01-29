#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import cv2
import numpy as np
import math
import typing
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy
import scipy.signal
from scipy.stats import linregress
from tqdm import tqdm
import re
import os, os.path
from os.path import splitext
import pydicom as dicom
import numpy as np
from pydicom.uid import UID, generate_uid
import shutil
from multiprocessing import dummy as multiprocessing
import time
import subprocess
import datetime
from datetime import date
import sys
import cv2
import matplotlib.pyplot as plt
import sys
from shutil import copy
import math
import torch
import torchvision
import echonet
import tqdm
from shutil import copyfile


def show(frame):
    cv2.imshow("test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def plot_point(frame,x,y,color=(0,255,0),radius = 0):
    thickness = -1
    return cv2.circle(frame, (x,y), radius, color, thickness)
def plot_line(frame,p1,p2,color=(0,191,255)):
    thickness = 2
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

        
def obtainContourPoints(img):
    # read image

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # set lower and upper bounds on blue color
    lower = (0,0,200)
    upper = (200,200,255)

    # threshold and invert so hexagon is white on black background
    thresh = cv2.inRange(rgb, lower, upper)
    # print(thresh.shape,np.min(thresh),np.max(thresh),type(thresh))
    # show(thresh)
    # get contours
    result = np.zeros_like(thresh)
    im2, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(im2,contours)
    #cv2.drawContours(img, im2, -1, (0,255,0), 3)

    im2 = np.array(im2)
    index_1 = 0
    index_2 = 0
    for i in im2:
        if i.shape[0]>im2[index_1].shape[0]:
            index_1=index_2
        index_2+=1
    rec = cv2.minAreaRect(im2[index_1])
    color = (0, 0, 255) 
    thickness = 2
    box = cv2.boxPoints(rec)
    box = np.int0(box)
    indexes = [0,1]
    for i in range(0,len(box)):
        for k in range(0,len(indexes)):
            if box[i][1]>box[indexes[k]][1] and not i in indexes:
                indexes[k]=i

    #cv2.drawContours(img,[box],0,(191,0,255),2)
    #img = plot_line(img,box[indexes[0]],box[indexes[1]])
    #img = plot_point(img,box[indexes[0],0],box[indexes[0],1])
    #img = plot_point(img,box[indexes[1],0],box[indexes[1],1])
    #cv2.rectangle(img,rec,color,thickness)
    return img, [box[indexes[0]].tolist(),box[indexes[1]].tolist()]

def calc_ratio(array,plot_dir,filename,vid=None,save = True):
    if save:
        plt.plot(array)
    x = scipy.signal.find_peaks(-np.array(array),distance=32)[0]
    ratios = []
    for i in range(0,len(x)):
        if save:
            plt.scatter(x[i],array[x[i]],color='orange')
        if i==len(x)-1:
            y = np.argmax(array[x[i]:])
        else:
            y = np.argmax(array[x[i]:x[i+1]])
        if save:
            plt.scatter(y+x[i],array[y+x[i]],color='red')
        x_val = array[x[i]]
        y_val = array[y+x[i]]
        ratios.append(x_val/y_val)
    if save:
        plt.savefig(os.path.join(plot_dir,filename[:-3]+'png'))
    ratios.sort()
    return np.mean(ratios[1:-1])


# In[2]:


def obtainThreshPoints(img,iterations = 1):
    # read image

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # set lower and upper bounds on blue color
    lower = (0,0,250)
    upper = (200,200,255)

    # threshold and invert so hexagon is white on black background
    thresh = cv2.inRange(rgb, lower, upper)
    

    # show(thresh)
    # get contours
    result = np.zeros_like(thresh)
    im2, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.dilate(thresh, None, iterations=iterations)
    #thresh = np.array([thresh,thresh,thresh])
    #thresh = np.transpose(thresh,(1,2,0))
    spare = thresh
    thresh = thresh==255

    img[thresh]=[255,255,255]

    #cv2.drawContours(img, im2, -1, (0,255,0), 3)

    im2 = np.array(im2)
    index_1 = 0
    index_2 = 0
    for i in im2:
        if i.shape[0]>im2[index_1].shape[0]:
            index_1=index_2
        index_2+=1
    rec = cv2.minAreaRect(im2[index_1])
    color = (0, 0, 255) 
    thickness = 2
    box = cv2.boxPoints(rec)
    box = np.int0(box)
    indexes = [0,1]
    for i in range(0,len(box)):
        for k in range(0,len(indexes)):
            if box[i][1]>box[indexes[k]][1] and not i in indexes:
                indexes[k]=i

    img = plot_point(img,box[indexes[0],0],box[indexes[0],1])
    img = plot_point(img,box[indexes[1],0],box[indexes[1],1])
    #cv2.rectangle(img,rec,color,thickness)
    #show(img)
    x1 = min(box[indexes[0],1],111)
    y1 = min(box[indexes[0],0],111)
    x2 = min(box[indexes[1],1],111)
    y2 = min(box[indexes[1],0],111)
    pair = (not thresh[x1,y1]) or (not thresh[x2,y2])
    return img,thresh, x1,y1,x2,y2, pair


# In[3]:


def midpoint(point1,point2):
    return (point1+point2)/2
def smooth(points):
    point_arr = [points[0]]
    for i in points[1:]:
        point_arr.append(midpoint(point_arr[-1],i))
    return np.array(point_arr)


# In[4]:


def get_points(vid):
    first = np.transpose(vid,(1,2,3,0))
    video = []
    guess = None
    vertexes = []
    for frame in range(0,len(first)):
        img,points = obtainContourPoints(first[frame])
        if np.linalg.norm(points[0])>np.linalg.norm(points[1]):
            c = points[0].copy()
            points[0] = points[1].copy()
            points[1] = c
        vertexes.append(points)

        # video.append(img)
        #show(img)

    #for point in vertexes:
    #change(np.array(vertexes))
    ok = np.array(vertexes)
    left_points = smooth(ok[:,0,:])
    right_points = smooth(ok[:,1,:])
    return left_points,right_points


# In[5]:


def get_dilation(vid,dilations = 1):
    frame_first_vid = np.transpose(vid,(1,2,3,0))
    threshes = []
    for frame in frame_first_vid:
        cv2_object = frame.copy()
        if dilations == 0:
            thresh = no_dilation(cv2_object,iterations=1)
        else:
            img,thresh,x1,y1,x2,y2,pair = obtainThreshPoints(cv2_object,iterations=dilations)
        
        threshes.append(thresh)
    
    return np.array(threshes)


# In[6]:


def strain_lengths(vid,threshes,first_points,second_points,filename,strain_dir,plot_dir,excel_dir):
    
    first = np.transpose(vid,(1,2,3,0))
    spare = first.copy()
    thresh = threshes
    frame_num = []
    x1s = []
    y1s = []
    error1 = []
    x2s = []
    y2s = []
    error2 = []
    length = []
    angle = []
    
    for frame in range(0,len(first)):
        rgb = cv2.cvtColor(first[frame,:,:], cv2.COLOR_BGR2RGB)
        t = (thresh[frame,:,:]*255).copy()
        t = cv2.inRange(t, 250, 255)
        im2, contours = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x1 = int(first_points[frame,0])
        y1 = int(first_points[frame,1])
        x2 = int(second_points[frame,0])
        y2 = int(second_points[frame,1])
        def calc_degrees(point):
            return math.atan2(point[1], point[0])
        deg = calc_degrees([x1-x2,y1-y2])
        angle.append(deg)
        final_point = im2[0][0][0]
        index = 0
        count = 0
        for j in im2[0]:
            count+=1
            if np.linalg.norm(j[0]-np.array([x1,y1]))< np.linalg.norm(final_point-np.array([x1,y1])):
                final_point = j[0]
                index = count
        
        final_point2 = im2[0][0][0]
        index2 = 0
        count2 = 0
        for j in im2[0]:
            count2+=1
            if np.linalg.norm(j[0]-np.array([x2,y2]))< np.linalg.norm(final_point2-np.array([x2,y2])):
                final_point2 = j[0]
                index2 = count2
        for k in range(0,index):
            plot_point(first[frame],im2[0][k][0][0],im2[0][k][0][1])
        for k in range(index2,len(im2[0])):
            plot_point(first[frame],im2[0][k][0][0],im2[0][k][0][1])
        for k in range(index,index2):
            plot_point(first[frame],im2[0][k][0][0],im2[0][k][0][1],color=(0,191,255))
        plot_point(first[frame],x1,y1,color=(255,0,255),radius=1)
        plot_point(first[frame],x2,y2,color=(255,0,255),radius=1)
        
        frame_num.append(frame)
        x1s.append(final_point[0])
        y1s.append(final_point[1])
        error1.append(np.linalg.norm(final_point-np.array([x1,y1])))
        x2s.append(final_point2[0])
        y2s.append(final_point2[1])
        error2.append(np.linalg.norm(final_point2-np.array([x2,y2])))
        length.append(len(im2[0])-index2+index)
    video = np.transpose(np.array(first),(3,0,1,2))

    savevideo(os.path.join(strain_dir,filename),video,fps=30)
    final = pd.DataFrame({'frame_num':frame_num,'x1':x1s,'y1':y1s,'error_1':error1,'x2':x2s,'y2':y2s,'error_2':error2,'length':length,'angle':angle})
    final.to_csv(os.path.join(excel_dir,filename[:-4]+'.csv'))
    return final,calc_ratio(length,plot_dir,filename)


# In[7]:


def estimate_strain(input_vid,weights,segmentation_dir,strain_dir,plot_dir,excel_dir,dilations = 1,segmenter = None,flip=True):
    if segmenter is None:
        segmenter = Segmentation(weights)
    
    video_file = segmenter.single_vid_prediction(input_vid,segmentation_dir,flip=flip)
    loaded_vid = loadvideo(video_file)
    left,right = get_points(loaded_vid)
    thresh = get_dilation(loaded_vid,dilations = dilations)
    measure = strain_lengths(loaded_vid,thresh,left,right,os.path.basename(input_vid),strain_dir,plot_dir,excel_dir)
    return measure[1]


# In[8]:




def collate_fn(x):
    x, f = zip(*x)
    i = list(map(lambda t: t.shape[1], x))
    x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
    return x, f, i


# In[9]:


class Segmentation:
    def __init__(self, segmentation_model_checkpoint, mean = np.array([31.834011, 31.95879,  32.082172]) , std = np.array([48.866325, 49.137333, 49.361984])):
        self.mean = mean
        self.std = std
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, aux_loss = False)
        self.model.classifier[-1] = torch.nn.Conv2d(self.model.classifier[-1].in_channels, 1, kernel_size = self.model.classifier[-1].kernel_size)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(device)
            checkpoint = torch.load(segmentation_model_checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()
    def single_vid_prediction(self,vid,output_folder,flip=True):
        device = torch.device("cuda")
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
        copyfile(vid, 'temp'+'\\'+vid.split('\\')[-1])
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
                newvideo[:,2,:,:] = np.maximum(newvideo[:,2,:,:], 255. * (y[:, 0, :, :] > 0))


                for (filename, offset) in zip(f,i):
                    #print(filename, offset, y[start:(start+offset), :, :, :].shape, x[start:(start+offset), :, :, :].shape)
                    np.save(os.path.join(output_folder, os.path.splitext(filename)[0]), y[start:(start+offset), 0, :, :])

                    #plain videos
                    echonet.utils.savevideo(os.path.join(output_folder,os.path.splitext(filename)[0] + ".avi"), np.transpose(newvideo[start:(start+offset), :, :, :],(1,0,2,3)).astype(np.uint8), 50)
                    
                    shutil.rmtree('temp')
                    return os.path.join(output_folder,os.path.splitext(filename)[0] + ".avi")






