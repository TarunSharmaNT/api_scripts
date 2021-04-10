###########################################
import os 
import torch
import psutil
print(os.getpid())
###########################################

###########################################
import socket
print(socket.gethostname())
##########################################

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
import copy
import sys
import socket


hostname = socket.gethostname()

#getting the parameters 
args = sys.argv

index = int(args[1])

start_index = int(args[2])

end_index = int(args[3])

dest = "/home/tarun/Number_Theory/Filtered_images/phase_2_testing/rear_door_l/dest_folder/"

source = "/home/tarun/Number_Theory/Filtered_images/phase_2_testing/rear_door_l/images/"

if hostname == "tarun-Lenovo-V14-IIL":
    source = "/home/tarun/Number_Theory/Filtered_images/phase_2_testing/rear_door_l/images/"
print(source)
    	
#selected_codes_for_augpoc = [
#	#"100", "102", "103", "105", "109", "110"
#        "109"
#]


#############################Global Variables###############################
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

#location code to defect loc code mapping

locationCode_locationCodeName_dict = {'BACKGROUND': 'BACKGROUND', 'Trunk Compartment': 'TRUNK', "Vehicle's Roof": 'ROOF', 'Engine Compartment': 'ENGINE', 'VIN': 'VIN', 'QR_Code': 'QR', 'Tyre': 'TYRE', 'Rear ¾ View Driver Side': 'REAR_45_LEFT', 'Front ¾ View Driver Side': 'FRONT_45_LEFT', 'Front ¾ View Passenger Side': 'FRONT_45_RIGHT', 'Rear ¾ View Passenger Side': 'REAR_45_RIGHT', 'Side View Driver Side': 'DRIVER_SIDE', 'Side View Passenger Side': 'PASSENGER_SIDE', 'Front View': 'FRONT_VIEW', 'Rear View': 'REAR_VIEW', 'STEERING': 'STEERING_WHEEL', 'GEAR': 'GEARSHIFT', 'AIR_INTAKE': 'AIR_INTAKE', 'DASH': 'DASHBOARD', 'WINDSHIELD': 'WINDSHIELD', 'HEAD_LIGHT_RIGHT': 'HEAD_LIGHT_RIGHT', 'HEAD_LIGHT_LEFT': 'HEAD_LIGHT_LEFT', 'HOOD': 'HOOD', 'FRONT View WINDOW Passenger Side': 'FRONT_WINDOW_PSIDE', 'REAR View WINDOW Passenger Side': 'REAR_WINDOW_PSIDE', 'Driver Side FRONT View DOOR': 'DSIDE_FRONT_DOOR', 'Driver Side REAR View DOOR': 'DSIDE_REAR_DOOR', 'REAR View WINDOW Driver Side': 'REAR_WINDOW_DSIDE', 'FRONT View WINDOW Driver Side': 'FRONT_WINDOW_DSIDE', 'Passenger Side FRONT View DOOR': 'PSIDE_FRONT_DOOR', 'Passenger Side REAR View DOOR': 'PSIDE_REAR_DOOR'}

std_loc_code_to_base_loc_code = {'BACKGROUND': 'BACKGROUND', 'Trunk Compartment': 'TRUNK', "Vehicle's Roof": 'ROOF', 'Engine Compartment': 'ENGINE', 'VIN': 'VIN', 'QR_Code': 'QR', 'Tyre': 'TYRE', 'Rear ¾ View Driver Side': 'REAR_45_LEFT', 'Front ¾ View Driver Side': 'FRONT_45_LEFT', 'Front ¾ View Passenger Side': 'FRONT_45_RIGHT', 'Rear ¾ View Passenger Side': 'REAR_45_RIGHT', 'Side View Driver Side': 'DRIVER_SIDE', 'Side View Passenger Side': 'PASSENGER_SIDE', 'Front View': 'FRONT_VIEW', 'Rear View': 'REAR_VIEW', 'STEERING': 'STEERING_WHEEL', 'GEAR': 'GEARSHIFT', 'AIR_INTAKE': 'AIR_INTAKE', 'DASH': 'DASHBOARD', 'WINDSHIELD': 'WINDSHIELD', 'HEAD_LIGHT_RIGHT': 'HEAD_LIGHT_RIGHT', 'HEAD_LIGHT_LEFT': 'HEAD_LIGHT_LEFT', 'HOOD': 'HOOD', 'FRONT View WINDOW Passenger Side': 'FRONT_WINDOW_PSIDE', 'REAR View WINDOW Passenger Side': 'REAR_WINDOW_PSIDE', 'Driver Side FRONT View DOOR': 'DSIDE_FRONT_DOOR', 'Driver Side REAR View DOOR': 'DSIDE_REAR_DOOR', 'REAR View WINDOW Driver Side': 'REAR_WINDOW_DSIDE', 'FRONT View WINDOW Driver Side': 'FRONT_WINDOW_DSIDE', 'Passenger Side FRONT View DOOR': 'PSIDE_FRONT_DOOR', 'Passenger Side REAR View DOOR': 'PSIDE_REAR_DOOR'}



#current time using the datetime library 
ct = (datetime.datetime.now())
timestamp = str(ct.timestamp())

#getting the date string 
date_ = str(ct).split(" ")[0].replace('-','_')

#similarly the time_ string 
time_ = timestamp.split(".")[0]

#forming the main time stamp
main_time_stamp = date_+time_


##########################Empty lists For the DataFrame Preparation##################
all_image_name = []
image_name_list = []
pred_location_code_list = []
pred_defect_location_code_list = []
pred_x_min_list = []
pred_y_min_list = []
pred_x_max_list = []
pred_y_max_list = []
pred_blur_val_list = []
pred_glare_val_list = []
pred_lowlight_val_list = []
pred_isinframe_val_list = []
pred_iscenteralign_val_list = []
pred_conf_val_list = []
pred_is_too_far_list=[]
pred_is_too_close_list = []
######################################################################################

###############################################################################################
count = 0
skipped = 0

print("processing image from {} to {}".format(start_index,end_index))
print("total images :", end_index-start_index)

free_gpu_list=[]
free_ram_list=[]
image_name_list=[]
inference_time_list=[]
for img in glob.glob(source+"/*.jpg")[start_index : end_index]:
			count=count+1
			if count%100 == 0:
				print("finished {} images".format(count))
				print("skipped {} images".format(skipped))
		#try:
			image_name = img.split("/")[-1]

			#First skipping is when we have different loc code other than we have chosen 	
			#if image_name.split('_')[2] not in selected_codes_for_augpoc:
			#skipped+=1
			#	continue
				

			#only code we have to change for different location code such as "windshield" or "window_front_right"
			#cmd = '''(echo -n '{ "tenant_id":"001","code":"windshield", "raw_image": "'; base64 ''' +img+'''; echo '"}') | curl -X POST "http://0.0.0.0:8001/input/" -H "Content-Type: application/json" -d @-'''
			cmd1 = '''(echo -n '{ "tenant_id":"001","raw_image": "'; base64 '''+img+'''; echo '"}') | curl -X POST "http://0.0.0.0:8000/CarImageStandardization/" -H "Content-Type: application/json" -d @-'''
			cmd2 = '''(echo -n '{ "tenant_id":"001","raw_image": "'; base64 '''+img+'''; echo '"}') | curl -X POST "http://0.0.0.0:8000/LocationCodeIdentification/" -H "Content-Type: application/json" -d @-'''
			cmd3 = '''(echo -n '{ "tenant_id":"001","raw_image": "'; base64 '''+img+'''; echo '"}') | curl -X POST "http://0.0.0.0:8000/CoordinatesIdentification/" -H "Content-Type: application/json" -d @-'''

			#exit()        
		
			#First APi
			try:
				###########################inference time#########################
				start=time.time()
				status1, output1 = subprocess.getstatusoutput(cmd1)
				end_time=time.time()-start
				inference_time_list.append(end_time)
				###################################################################

				########free gpu mem computation################
				free_gpu_temp = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0))/2.**30
				free_gpu_list.append(free_gpu_temp)
				################################################


				########free cpu computation####################
				free_ram_temp = psutil.virtual_memory().free/2.**30
				free_ram_list.append(free_ram_temp)
				####################################################

				try:
					main_output1 = output1.split("\n")[-1].replace('false','False')
					main_output1 = main_output1.split("\n")[-1].replace('true','True')
					main_output1 = eval(main_output1)
				except:
					continue
				detection_status_1 = main_output1["detection_status"]
				if detection_status_1 == True:
					
					num_of_detections = main_output1["number_of_detections"]

					standardization_result = main_output1["CarImageStandardization_result"]

					for i in range(num_of_detections):
						str_req = "detection_"+str(i+1)
						is_too_far = standardization_result[str_req]["is_too_far"]
						is_too_close = standardization_result[str_req]["is_too_close"]

						blur_validation = standardization_result["blur_validation"]
						glare_validation = standardization_result["glare_light_validation"]
						low_light_validation = standardization_result["low_light_validation"]

						all_image_name.append(image_name)
						pred_lowlight_val_list.append(low_light_validation)
						pred_glare_val_list.append(glare_validation)
						pred_blur_val_list.append(blur_validation)
						pred_is_too_far_list.append(is_too_far)
						pred_is_too_close_list.append(is_too_close)				
				else:
					all_image_name.append(image_name)
					pred_lowlight_val_list.append("dummy")
					pred_glare_val_list.append("dummy")
					pred_blur_val_list.append("dummy")
					pred_is_too_far_list.append("dummy")
					pred_is_too_close_list.append("dummy")




			except:
					print("Exception 1 #####################################################")
					print("this will never execute because the errors are handled already")
					exit()
					blur_validation = "dummy"
					glare_validation = "dummy"
					low_light_validation = "dummy"
						
					all_image_name.append(image_name)
					pred_lowlight_val_list.append("dummy")
					pred_glare_val_list.append("dummy")
					pred_blur_val_list.append("dummy")
					pred_is_too_far_list.append("dummy")
					pred_is_too_close_list.append("dummy")
					#continue       

			try:
				#Second Api
				status2, output2 = subprocess.getstatusoutput(cmd2)

				main_output2 = output2.split("\n")[-1].replace('false','False')
				main_output2 = main_output2.split("\n")[-1].replace('true','True')
				main_output2 = eval(main_output2)

				###########it will handle all cases################## 
				detection_status_2 = main_output2["detection_status"]
				if detection_status_2 == True:
					num_of_detections = main_output2["number_of_detections"]
					for i in range(num_of_detections):
						str_req = "detection_" + str(i+1)
						location_code = main_output2["LocationCodeIdentification_result"][str_req]["location_code"]
						conf = main_output2["LocationCodeIdentification_result"][str_req]["confidence"]
						location_code = std_loc_code_to_base_loc_code[location_code]
						
						pred_location_code_list.append(location_code)  
						pred_conf_val_list.append(conf) 

				else:
					location_code = "dummy"
					conf = "dummy"
					pred_location_code_list.append(location_code)  
					pred_conf_val_list.append(conf) 
				########################################################    

			except:
				print("Exception 2")
				exit()
				location_code = "dummy"
				conf = "dummy"
				pred_location_code_list.append(location_code)  
				pred_conf_val_list.append(conf) 
				#continue

			try:    
				#Third Api
				status3, output3 = subprocess.getstatusoutput(cmd3)
				#print("*****",status)
				#print("******",output)
				#print(output)
		
				#exit()
				#######
				main_output3 = output3.split("\n")[-1].replace('false','False')
				main_output3 = main_output3.split("\n")[-1].replace('true','True')
				main_output3 = eval(main_output3)

				detection_status_3 = main_output3["detection_status"]
				is_inframe = "dummy"
				is_center_align ="dummy"
				###########it will handle all the cases #######################
				###############################################################    
				if detection_status_3 == True:
					num_of_detections  = main_output3["number_of_detections"]   
					coordinates_result = main_output3["CoordinatesIdentification_result"]

					for i in range(num_of_detections):
						detection_name = "detection_"+str(i+1)
						#coordinates_result = coordinates_result[detection_name]
						bbox = coordinates_result[detection_name]["bounding_box"]
						pred_x_min_list.append(bbox['xmin'])
						pred_y_min_list.append(bbox['ymin'])
						pred_x_max_list.append(bbox['xmax'])
						pred_y_max_list.append(bbox['ymax'])

						ai_guidance = coordinates_result[detection_name]["ai_guidance"]

						if ai_guidance == True:
							is_inframe = coordinates_result[detection_name]["is_in_frame"]
							is_center_align = coordinates_result[detection_name]["is_center_alligned"]
						else:
							is_inframe = "dummy"
							is_center_align = "dummy"


						pred_isinframe_val_list.append(is_inframe)
						pred_iscenteralign_val_list.append(is_center_align)
					############################################################################
				else:
					bbox = "dummy"
					pred_x_min_list.append(bbox)
					pred_y_min_list.append(bbox)
					pred_x_max_list.append(bbox)
					pred_y_max_list.append(bbox)
					pred_isinframe_val_list.append("dummy")
					pred_iscenteralign_val_list.append("dummy")

			except:
				print("Exception 3")
				exit()
				bbox = "dummy"
				pred_x_min_list.append(bbox)
				pred_y_min_list.append(bbox)
				pred_x_max_list.append(bbox)
				pred_y_max_list.append(bbox)
				pred_isinframe_val_list.append("dummy")
				pred_iscenteralign_val_list.append("dummy")
				#continue




# print(count)

###############editing the Old DataFrame######################



# print(len(pred_location_code_list))
# print(len(pred_y_max_list))
# print(len(pred_iscenteralign_val_list))
# print(len(pred_x_max_list))
# print(len(pred_isinframe_val_list))
# #exit("exit")


inference_df=pd.DataFrame({"image_name":image_name_list,"infer_time":inference_time_list,"free_ram":free_ram_list,"free_gpu":free_gpu_list})
inference_time_list.to_csv(index+".csv",index=False)


#First make a copy of the old dataframe######################################
new_df = pd.DataFrame()
new_df['image_name'] = all_image_name
new_df["pred_loc_code"] = pred_location_code_list
new_df["pred_confidence"] = pred_conf_val_list
new_df["pred_is_blur"] = pred_blur_val_list
new_df["pred_is_low_light"]=pred_lowlight_val_list
new_df["pred_is_glare"]=pred_glare_val_list
new_df["p_xmin"]=pred_x_min_list
new_df["p_ymin"]=pred_y_min_list
new_df["p_xmax"]=pred_x_max_list
new_df["p_ymax"]=pred_y_max_list
new_df["is_inframe"]=pred_isinframe_val_list
new_df["is_center_alligned"] = pred_iscenteralign_val_list
new_df["is_too_far"] = pred_is_too_far_list
new_df["is_too_close"] = pred_is_too_close_list
##############################################################################
# print(new_df.columns)
#exit()
######################freeing up the memory######################################
image_name_list = []
pred_location_code_list = []
pred_defect_location_code_list = []
pred_x_min_list = []
pred_y_min_list = []
pred_x_max_list = []
pred_y_max_list = []
pred_blur_val_list = []
pred_glare_val_list = []
pred_lowlight_val_list = []
pred_isinframe_val_list = []
pred_iscenteralign_val_list = []
pred_conf_val_list = []
pred_is_too_far_list = []
pred_is_too_close_list=[]
#################################################################################





##########################Saving up the csv#########################################
#new_df.to_csv(dest+"/"+main_time_stamp+ "_pred_values_"+str(index)+"_.csv",index=False)
#print("finish")
#exit()
###################################################################################
