##########################Author Tarun Sharma ###############################
##########################Dated March 26#####################################


#################################
import os
print(os.getpid())


import socket
print(socket.gethostname())
#################################

import shutil
import glob
import copy
import pandas as pd
import numpy as np

#importing the cv2
import cv2



hostname=socket.gethostname()

source_folder =""
csv_path=""
dest_folder=""
if hostname == "tarun-Lenovo-V14-IIL":
    source_folder =""
    csv_path=""
    dest_folder=""

#reading the csv
df = pd.read_csv("")


#image_name_list
image_name_list = df["image_name"].tolist()

for img in glob.glob():
    image_name = img.split("/")[-1]


    if image_name not in image_name_list:
        continue


    #copy for the 
    shutil.copy(img,dest_folder)

    




