# Strain Analysis

This project documents the process to obtain deep learning derived LV global longitudinal strain from an apical-4-chamber echocardiogram video.

## Setup
1. Get the LV segmentation model (deeplabv3_resnet50_random.pt) weights from EchoNet-Dynamic, or train your own LV segmention model using videos from https://github.com/echonet/dynamic
2. place all the strain videos as .avi files in a single folder

## Processing

You may now run the code using the Script Test (Ishan).ipynb notebook.

## Output
There are 4 relivent folders in the output directory, and one csv.
1. output/Segmentation is the folder that contains the videos of the segmentation. If something went wrong, check here, as echonet may not be processing the videos correctly
2. output/Strain is the folder that contains the outline of the strain overlayed on the echocardiogram.
3. output/Plot is the folder that contains plots of the measurement of length in each frame
4. output/Excel is the folder that contains the measurement of length in each frame
5. output/Measured Strain.csv is the file that contains the mean estimate of strain for each video. These estimates can be converted to traditional measurements of strain using 100*(1-estimate)
