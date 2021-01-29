import os
from Vid_to_Strain import estimate_strain
from tqdm import tqdm
from datetime import datetime




folder = 'D:\\Netherlands'
weights = 'C:\\Users\\Remote\\Documents\\John\\echonet_segmentation\\Weights-20201103T193519Z-001\\Weights\\deeplabv3_resnet50_random.pt'
segmentation_dir = 'C:\\Users\\Remote\\Documents\\John\\echonet_segmentation_analysis\\Segmentation Analysis\\Netherlands\\Segmentation'
strain_dir = 'C:\\Users\\Remote\\Documents\\John\\echonet_segmentation_analysis\\Segmentation Analysis\\Netherlands\\Strain'
plot_dir = 'C:\\Users\\Remote\\Documents\\John\\echonet_segmentation_analysis\\Segmentation Analysis\\Netherlands\\Plot'
excel_dir = 'C:\\Users\\Remote\\Documents\\John\\echonet_segmentation_analysis\\Segmentation Analysis\\Netherlands\\Excel'




for vid in tqdm(os.listdir(folder)):
    filename = []
    time = []
    if vid[-4:] == '.avi':
        start = datetime.now()
        estimate_strain(folder+'\\'+vid,weights,segmentation_dir,strain_dir,plot_dir,excel_dir)
        time.append(datetime.now()-start)
        filename.append(vid)