#gunicorn -k uvicorn.workers.UvicornWorker --bind "0.0.0.0:8001" --log-level debug main:app
#################Edited March 26 Tarun Sharma ################################
##############################################################################

from resource import error
import numpy as np
import pandas as pd
import glob
import subprocess
import os
from subprocess import check_output
from shlex import split
import shlex
import subprocess
import time
import datetime
import cv2
import matplotlib.pyplot as plt
import sys
import socket

##########################################
print("Print Process id ")
print(os.getpid())
###########################################
print("hostname")
hostname = socket.gethostname()
print(hostname)
###########################################
argv = sys.args
index = argv[1]
start_index = argv[2]
end_index = argv[3]
################################## PARAMS
image_source_folder = "/home/tarun/Number_Theory/New_Data_Preparation/Rear_door/five_thousand/user_6/"

std_api_result_csv = "/home/tarun/Number_Theory/New_Data_Preparation/Rear_door/five_thousand/std_api/main.csv"

csv_destination = "/home/tarun/Number_Theory/New_Data_Preparation/Rear_door/five_thousand/defect_csv/"
###################
try:
    df = pd.read_csv(std_api_result_csv)
except:
    print("file not found")

ct = (datetime.datetime.now())
timestamp = str(ct.timestamp())
date_ = str(ct).split(" ")[0].replace('-','_')
time_ = timestamp.split(".")[0]

main_time_stamp = date_ + time_

csv_name = main_time_stamp + "defect_api_result.csv"
#######################################################################################

all_image_names = []
all_defect_location_name = []
all_detection_status = []
all_defect_type = []
all_xmin = []
all_ymin = []
all_xmax = []
all_ymax = []


body_parts = ['front_door_right','front_door_left',"rear_door_right", "rear_door_left", "hood"]

glass_parts = ["window_front_right", "window_front_left", "window_rear_left", "window_rear_right","windshield"]

map_defect_location_code = {
    'PSIDE_FRONT_DOOR':'front_door_right',
    'DSIDE_FRONT_DOOR':'front_door_left',
    'PSIDE_REAR_DOOR':"rear_door_right", 
    'DSIDE_REAR_DOOR':"rear_door_left", 
    'HOOD':"hood",
    'FRONT_WINDOW_PSIDE':"window_front_right", 
    'FRONT_WINDOW_DSIDE':"window_front_left", 
    'REAR_WINDOW_DSIDE':"window_rear_left", 
    'REAR_WINDOW_PSIDE':"window_rear_right",
    'WINDSHIELD':"windshield"
}


for img in glob.glob(image_source_folder + "/*.jpg")[start_index:end_index]:

        image_name = img.split('/')[-1]
#        image_name = image_name.replace(".jpg","_hood.jpg")
        print(image_name)
        
        if image_name not in df['image_name'].tolist():
            print("image not present STD result api")
            continue

        sub = df.loc[df['image_name']==image_name,]


        for i in range(len(sub)):
            temp_df = sub.iloc[i,]

            location_code = temp_df['pred_loc_code']
            
            # check if key exists else pass
            if location_code not in map_defect_location_code.keys():
                print("location not valid for defect detection")
                continue

            defect_location_code = map_defect_location_code[location_code]

            if defect_location_code == "dummy" or defect_location_code == "no_detection":
                continue

            xmin = int(temp_df['p_xmin'])
            ymin = int(temp_df['p_ymin'])
            xmax = int(temp_df['p_xmax'])
            ymax = int(temp_df['p_ymax'])


            cmd = '''(echo -n '{ "tenant_id":"001","code":"''' + defect_location_code + '''", "xmin":"''' + str(xmin) + '''", "ymin":"''' + str(ymin) + '''", "xmax":''' + str(xmax) + ''',"ymax":''' + str(ymax) + ''', "raw_image": "'; base64 ''' + img + '''; echo '"}') | curl -X POST "http://0.0.0.0:8001/DefectIdentification/" -H "Content-Type: application/json" -d @-'''        
            start=time.time()
            print("Start time:::",start)
            status, output = subprocess.getstatusoutput(cmd)
            print("end time:::",time.time()-start)
            main_output = output.split("\n")[-1].replace('false','False')
            main_output = main_output.split("\n")[-1].replace('true','True')
            main_output = eval(main_output)
            
            try:
                result = main_output['result']
            except:
                print("Error status from server side")
                continue

            is_defect_found = result['is_defect_found']

            if is_defect_found == True:
                if defect_location_code in body_parts:
                        
                    dent_coordinates = result['dent_coordinates']
                    scratch_coordinates = result['scratch_coordinates']
                        
                    for i in range(len(dent_coordinates)):
                        temp_dict = dent_coordinates[i]
                        xmin = temp_dict['xmin']
                        ymin = temp_dict['ymin']
                        xmax = temp_dict['xmax']
                        ymax = temp_dict['ymax']
 
                        all_image_names.append(image_name)
                        all_defect_location_name.append(defect_location_code)
                        all_detection_status.append("True")
                        all_defect_type.append("dent_coordinates")
                        all_xmin.append(xmin)
                        all_ymin.append(ymin)
                        all_xmax.append(xmax)
                        all_ymax.append(ymax)

                    for i in range(len(scratch_coordinates)):
                        temp_dict = scratch_coordinates[i]
                        xmin = temp_dict['xmin']
                        ymin = temp_dict['ymin']
                        xmax = temp_dict['xmax']
                        ymax = temp_dict['ymax']

                        all_image_names.append(image_name)
                        all_defect_location_name.append(defect_location_code)
                        all_detection_status.append("True")
                        all_defect_type.append("scratch_coordinates")
                        all_xmin.append(xmin)
                        all_ymin.append(ymin)
                        all_xmax.append(xmax)
                        all_ymax.append(ymax)

                elif defect_location_code in glass_parts:

                    defect_coordinates = result['defect_coordinates']
                        
                    for i in range(len(defect_coordinates)):
                        temp_dict = defect_coordinates[i]
                        xmin = temp_dict['xmin']
                        ymin = temp_dict['ymin']
                        xmax = temp_dict['xmax']
                        ymax = temp_dict['ymax']
 
                        all_image_names.append(image_name)
                        all_defect_location_name.append(defect_location_code)
                        all_detection_status.append("True")
                        all_defect_type.append("defect_coordinates")
                        all_xmin.append(xmin)
                        all_ymin.append(ymin)
                        all_xmax.append(xmax)
                        all_ymax.append(ymax)
                else:
                    pass
            else :
                all_image_names.append(image_name)
                all_defect_location_name.append(defect_location_code)
                all_detection_status.append("False")
                all_defect_type.append("dummy")
                all_xmin.append("dummy")
                all_ymin.append("dummy")
                all_xmax.append("dummy")
                all_ymax.append("dummy")
                    


temp = pd.DataFrame()


temp['image_name'] = all_image_names
temp['defect_location_code'] = all_defect_location_name
temp['detection_status'] = all_detection_status 
temp['defect_type'] = all_defect_type
temp['xmin'] = all_xmin
temp['ymin'] = all_ymin 
temp['xmax'] = all_xmax
temp['ymax'] = all_ymax

temp.to_csv(csv_destination + csv_name, index=False)


