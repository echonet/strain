import os
from Vid_to_Strain import estimate_strain
from Vid_to_Strain import Segmentation
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Produce Strain Calculation for folder.')
parser.add_argument("-i",'--input_folder', help='Folder of echonet videos')
parser.add_argument('-w','--model_weights', help='echonet deeplabv3_resnet50_random.pt weights')

parser.add_argument("-s",'--window_size', default=3, help='size of moving average window')
parser.add_argument('-d','--dilations', default=3, help='number of dilations to perform')
parser.add_argument("-h",'--downsample', default=3, help='downsample rate')
parser.add_argument('-t','--thickness', default=1, help='line thickness')
parser.add_argument('-f','--flip', default=1, help='verticle axis flip')




args = parser.parse_args()

weights = args.model_weights
segmenter = Segmentation(weights)
window_size = args.window_size
dilations = args.dilations
downsample = args.downsample
thickness = args.thickness
flip = args.flip


folder = args.input_folder
output = 'output'
try:
    os.mkdir(output)
except FileExistsError:
    print("exists")
segmentation_dir = os.path.join(output, 'Segmentation')
strain_dir = os.path.join(output, 'Strain')
plot_dir = os.path.join(output, 'Plot')
excel_dir = os.path.join(output, 'Excel')
try:
    os.mkdir(segmentation_dir)
except FileExistsError:
    print("exists")
try:
    os.mkdir(strain_dir)
except FileExistsError:
    print("exists")
try:
    os.mkdir(plot_dir)
except FileExistsError:
    print("exists")
try:
    os.mkdir(excel_dir)
except FileExistsError:
    print("exists")

filename = []
strain = []
mean_strain = []
for vid in tqdm(os.listdir(folder)):#['2PXH1WVX_4_EPIQ7C_NO.avi']:
    if vid[-4:] == '.avi':
        try:
            start = datetime.now()
            output_filename = vid[:-4]+'_dilation'+str(dilations)+'_downsample'+str(downsample)+"_thickness"+str(thickness)+".avi"
            measured_strain = estimate_strain(os.path.join(folder,vid),weights,segmentation_dir,strain_dir,plot_dir,excel_dir,segmenter=segmenter,window_size=window_size,flip=flip,dilations=dilations, downsample = downsample,output_filename=output_filename)
            mean_strain.append(np.mean(measured_strain))
            filename.append(vid)
        except:
            print(vid)
pd.DataFrame({"Filename":filename, 'Mean Strain':mean_strain}).to_csv(os.path.join(output,"Measured Strain.csv"),index=False)