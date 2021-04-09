#nohup sudo /home/nt/anaconda3/envs/pytorch17/bin/python -u process_defect_for_hood_feb26.py > process_defect_for_hood_feb26.log 2>&1&

##################Author Tarun Sharma ###########################
##################Dated March 19#################################

#So it will handle the new plan that we are using that is looking into the defect and non defect images
#importing the libraries 
from pathlib import Path
from numpy.lib.function_base import append
import pandas as pd 
import glob 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import copy
import os
import socket

from typing_extensions import final
#############################################################
process_id = os.getpid()
print(process_id)
hostname = socket.gethostname()
print(hostname)
#############################################################

def get_unique_image_names_based_on_labels(df, label):
	"""
		From the csv collect image_names with required annotation 
		and save as csv,
		we can use this csv to to get/copy missing images
	""" 
	df = df.loc[df["region_attributes"]==label,]
	temp= pd.DataFrame()
	temp['image_name'] = df['filename'].unique().tolist()
	temp.to_csv("missing_hood_feb25.csv", index =False)
	del temp


#getting all the available images from list of dir
def get_available_image_names_from_list_of_dir(image_folder_paths):
	"""
		validate available images and image_names present in csv 
	"""
	all_available_images = []

	for folder in image_folder_paths:
		all_available_images +=[ img.split('/')[-1] for img in glob.glob(folder+"/*.jpg")]

	#list of all available images
	return all_available_images



def filter_csv_with_required_labels(df, req_tags):
	"""
		Filter csv with required labels
	"""
	return df.loc[df['region_attributes'].isin(req_tags),]

def make_sure_all_defect_annotation_has_segmentation_mask(df, segmentation_label):
	"""
		check for all dent/scratch a segmentation mask is there.
		This is not a valid sanity check for some cases
	"""
	all_hood_img = df.loc[df['region_attributes']==segmentation_label,'filename'].unique().tolist()
	df = df.loc[df['filename'].isin(all_hood_img),]
	df = df.loc[df['filename'].isin(all_available_images),]

	return df




def read_all_csv_from_dir_list(folder_paths):
	"""
		read all csv from given list of dir
	""" 
	'''
	#Old way of reading the csvs
	#df = pd.DataFrame()
	#for folder in folder_paths:
	#    for csv in glob.glob(folder+"/*.csv"):
	#        temp = pd.read_csv(csv)
	#        df = pd.concat([df,temp], axis=0)
	'''
	df = pd.concat([pd.read_csv(path) for path in Path(folder_paths).rglob('*.csv')],axis=0)
	return df

if __name__ == "__main__":
	################Remember to handle the defect properly that is the main priority ###################

	#the images from which we prepare the data fot the annotation guys
	##############################################################################################
	main_concatenated_csv = "/home/tarun/Number_Theory/New_Data_Preparation/Rear_door/csv_folder/final.csv"	
	old_df  = pd.read_csv(main_concatenated_csv)
	old_df = old_df[["image_name","pred_loc_code"]]
	print(old_df.columns)
	#exit("first debug point")
	
	###############################################################################################
	

	
	
	#Only Parent folder is needed no need to loop to further directories the code handles inside directories
	##########################################csv Folders##########################################
	Parent_folder_for_each_intern ="/home/nt/tarun_sharma/image_explore_2/backup_images_for_processing/front_view_images_feb28_splitted/Annotation_history/"
	if hostname == "tarun-Lenovo-V14-IIL":
		Parent_folder_for_each_intern = "/home/tarun/Number_Theory/Annotation_history/Data/" 
	'''	
	folder_paths_csvs=[]
	for folder in os.listdir(Parent_folder_for_each_intern):
		if os.path.isdir(Parent_folder_for_each_intern + folder):
			folder_paths_csvs.append(folder)
	'''
	###############################################################################################
	
	
	#########################################images Folders########################################
	Parent_folder_images = "/home/nt/tarun_sharma/image_explore_2/backup_images_for_processing/front_view_images_feb28_splitted/done_images/"
	if hostname == "tarun-Lenovo-V14-IIL":
		Parent_folder_images = "/home/tarun/Number_Theory/New_Data_Preparation/Validation_csv/images_door/" 

	folder_paths_images = []
	for folder in os.listdir(Parent_folder_images):
		if os.path.isdir(Parent_folder_images + folder):
			folder_paths_images.append(Parent_folder_images+folder)
	###############################################################################################

	#image_list_from_main_csv = [path.split("/")[-1] for path in Path(Parent_folder_images).rglob('*.jpg')]
	
	
	#This we need to update now 
	#require tags to filter from the region_attributes
	req_tags = ['{"labels":"door"}','{"labels":"window"}','{"labels":"HOOD"}','{"labels":"WINDSHIELD"}','{"labels":"dent"}','{"labels":"scratch"}','{"labels":"defect"}','{"labels":"FRONT_VIEW"}','{"labels":"REAR_VIEW"}','{"labels":"REAR_45_LEFT"}','{"labels":"FRONT_45_LEFT"}','{"labels":"FRONT_45_RIGHT"}','{"labels":"REAR_45_RIGHT"}','{"labels":"DRIVER_SIDE"}','{"labels":"PASSENGER_SIDE"}','{"labels":"STEERING_WHEEL"}','{"labels":"GEARSHIFT"}','{"labels":"AIR_INTAKE"}','{"labels":"DASHBOARD"}','{"labels":"HEAD_LIGHT_RIGHT"}','{"labels":"HEAD_LIGHT_LEFT"}','{"labels":"FRONT_WINDOW_PSIDE"}','{"labels":"DSIDE_FRONT_DOOR"}','{"labels":"DSIDE_REAR_DOOR"}','{"labels":"REAR_WINDOW_DSIDE"}','{"labels":"PSIDE_FRONT_DOOR"}','{"labels":"PSIDE_REAR_DOOR"}','{"labels":"FRONT_WINDOW_DSIDE"}','{"labels":"REAR_WINDOW_PSIDE"}']

	#reading all the csvs from the folder paths
	df = read_all_csv_from_dir_list(Parent_folder_for_each_intern)


	#destination of the defect images
	dest_defect_image = "/home/nt/tarun_sharma/image_explore_2/backup_images_for_processing/front_view_images_feb28_splitted/defect_upto_march12/"
	if hostname == "tarun-Lenovo-V14-IIL":
		dest_defect_image = "/home/tarun/Number_Theory/Annotation_history/defect_upto_march_28_rear_door/defect/"
	##############################################################################################################################


	################non defect images destination path #########################################################
	dest_non_defect_image = "/home/tarun/Number_Theory/New_Data_Preparation/Processing_code/Data/Non_Defect/"
	if hostname == "tarun-Lenovo-V14-IIL":
			dest_non_defect_image = "/home/tarun/Number_Theory/Annotation_history/defect_upto_march_28_rear_door/non_defect/"
	############################################################################################################


	#printing the columns 
	print(df.columns)
	#printing the dataframe shape
	print(df.shape)

	#printing the region_attributes count
	print(df['region_attributes'].value_counts())
	
	#you can check the value count by uncommenting the exit line
	#exit()

	#get image names for hood annotation as csv
	#get_unique_image_names_based_on_labels(df, '{"labels":"hood"}') 
	#exit()

	# get available image names from dir_list
	all_available_images = get_available_image_names_from_list_of_dir(folder_paths_images)
	print(len(all_available_images))

	#exit("first debug")

	#filter the csv with available images 
	df = df.loc[df['filename'].isin(all_available_images),]
	print(df.shape)
	print(df['region_attributes'].value_counts())
	
	#Filter the csv with the required available tags that we have
	df = filter_csv_with_required_labels(df, req_tags)
	print(df.shape)
	print(df['region_attributes'].value_counts())


	#this is not a valid checking for all cases
	#df = make_sure_all_defect_annotation_has_segmentation_mask(df, segmentation_label='{"labels":"hood"}')
	#print(df.shape)
	#print(df['region_attributes'].value_counts())
	#exit("finish")
	#############################################################################
	list_of_unique_images = df['filename'].unique().tolist()
	#############################################################################
	#Correct way to get rid of the images that they skipped 
	image_not_need_to_consider = list(set(image_list_from_main_csv) - set(list_of_unique_images))
	#############################################################################

	######################Empty list######################################
	xmin_list = []
	ymin_list = []
	xmax_list = []
	ymax_list = []
	image_name_list = []
	label_list = []
	old_label_list=[]
	#######################################################################


	#iterating over all the image paths
	for folder in folder_paths_images:
		print(folder)
		for img in glob.glob(folder +'/*.jpg'):
				
				#print("HIII")
				# Take one image from the dir_list

				#image_name from the image_path
				image_name = img.split("/")[-1]
				
				
				if image_name not in list_of_unique_images or image_name in image_not_need_to_consider:
					continue
				
				print("HIII")
				#reading the image with try and catch as image is prone to error
				try:         
					image = cv2.imread(img)
				except:
					print("Problem in Reading the image")
					continue

				#create dummy channels
				# dummy channels should be in the shape of original image
				# All dummy mask created as a single channel image
				seg_mask = np.zeros([image.shape[0],image.shape[1],1],dtype=np.uint8)#segmentation mask
				bty_mask = np.zeros([image.shape[0],image.shape[1],1],dtype=np.uint8)#boundary mask
				dent_mask = np.zeros([image.shape[0],image.shape[1],1],dtype=np.uint8)#dent mask
				scratch_mask = np.zeros([image.shape[0],image.shape[1],1],dtype=np.uint8)#scratch mask


				#dummy channel for the window 
				seg_mask_window = np.zeros([image.shape[0],image.shape[1],1],dtype=np.uint8)
				bty_mask_window = np.zeros([image.shape[0],image.shape[1],1],dtype=np.uint8)
				defect_mask_window = np.zeros([image.shape[0],image.shape[1],1],dtype=np.uint8) 
				# create a sub-DataFrame for that selecte image_name
				#this data frame has one segmentation annotation, and n dent/scratch annotation
				#print(df.columns)
				#exit("first debug point")
				print(image_name)
				
				#print(df.head())
				#exit("finish")
				sub = df.loc[df["filename"] == image_name,]

				region_label_list = []

				#if there is a major issue then simply skip the image
				major_issue = False

				for i in range(len(sub)):
					# Take one row
					temp = sub.iloc[i,]

					#get the location name/ labels
					loc = temp['region_attributes']
					
					#appending into the region_label_list
					region_label_list.append(loc)

					#convert dict_as_string to dict
					data = temp['region_shape_attributes']
					data = eval(data)

					# combine list of x (eg:[1,2,3]) and list of y (eg:[4,5,6]) as list of poist, then :
					# z = [(1,4), (2,5), (3,6)]
					if loc in [ '{"labels":"door"}', '{"labels":"dent"}', '{"labels":"scratch"}','{"labels":"window"}','{"labels":"defect"}']: 
						list_of_keys = [*data.keys()]
						print(list_of_keys) 
						if 'x' in list_of_keys or 'y' in list_of_keys or 'height' in list_of_keys or 'width' in list_of_keys:
							major_issue = True
							break
						z = list(zip(data['all_points_x'], data['all_points_y']))
						contours = np.array([list(x) for x in z])
					else:
						list_of_keys = [*data.keys()]
						print(list_of_keys)
						if 'all_points_x' in list_of_keys or 'all_points_y' in list_of_keys:
							major_issue = True
							break


					# create mask for each label
					# this conditions change in diff cases
					if loc == '{"labels":"door"}':
						seg_mask = cv2.fillPoly( seg_mask, np.int32([contours]) , (255) )
						bty_mask = cv2.polylines(bty_mask, np.int32([contours]),  True, (255), 3) 
						# plt.imshow(seg_mask)
						# plt.show()
						# plt.imshow(bty_mask)
						# plt.show()
					elif loc == '{"labels":"dent"}' or loc == '{"labels":undefined}':
						dent_mask = cv2.fillPoly( dent_mask, np.int32([contours]) , (255) )
						# plt.imshow(dent_mask)
						# plt.show()
					elif loc == '{"labels":"scratch"}':
						scratch_mask = cv2.fillPoly( scratch_mask, np.int32([contours]) , (255) )
						# plt.imshow(scratch_mask)
						# plt.show()
					elif loc ==  '{"labels":"window"}':
						seg_mask_window= cv2.fillPoly(seg_mask_window, np.int32([contours]) , (255) )
						bty_mask_window = cv2.polylines(bty_mask_window, np.int32([contours]),  True, (255), 3) 
					elif loc == '{"labels":"defect"}':
						defect_mask_window = cv2.fillPoly( defect_mask_window, np.int32([contours]) , (255) )
					elif loc in '{"labels":"WINDSHIELD}' or loc in '{"labels":"HOOD"}' or loc in '{"labels":"FRONT_VIEW"}' or loc in '{"labels":"REAR_VIEW"}' or loc in '{"labels":"REAR_45_LEFT"}' or loc in '{"labels":"FRONT_45_LEFT"}' or loc in '{"labels":"FRONT_45_RIGHT"}' or loc in '{"labels":"REAR_45_RIGHT"}' or loc in '{"labels":"DRIVER_SIDE"}' or loc in '{"labels":"PASSENGER_SIDE"}' or loc in '{"labels":"STEERING_WHEEL"}' or loc in '{"labels":"GEARSHIFT"}' or loc in '{"labels":"AIR_INTAKE"}' or loc in '{"labels":"DASHBOARD"}' or loc in '{"labels":"HEAD_LIGHT_RIGHT"}' or loc in '{"labels":"HEAD_LIGHT_LEFT"}' or loc in '{"labels":"FRONT_WINDOW_PSIDE"}' or loc in '{"labels":"DSIDE_FRONT_DOOR"}' or loc in '{"labels":"DSIDE_REAR_DOOR"}' or loc in '{"labels":"REAR_WINDOW_DSIDE"}' or loc in '{"labels":"PSIDE_FRONT_DOOR"}' or loc in '{"labels":"PSIDE_REAR_DOOR"}' or loc in '{"labels":"FRONT_WINDOW_DSIDE"}' or loc in '{"labels":"REAR_WINDOW_PSIDE"}':
						xmin = int(data['x'])
						ymin = int(data['y'])
						xmax = int(data['width']) + xmin
						ymax = int(data['height']) + ymin


						old_label = old_df.loc[old_df["image_name"]==image_name,]
						print(old_label.shape)
						#exit("second debug point")

						xmin_list.append(xmin)
						ymin_list.append(ymin)
						xmax_list.append(xmax)
						ymax_list.append(ymax)
						image_name_list.append(image_name)
						old_label_list.append(old_label)
						
												
						#evaluating the loc codes
						loc_code = eval(loc)
						loc_code = loc_code["labels"]
						
						label_list.append(loc_code)
					
				
				
				if major_issue:
    					continue				
				# if any error in segmentation annotation, eg case : no seg mask after fillpoly
				#then skip those image 
				#this validation is not valid for some cases
				
				#in this scenerio we have to consider other cases also
				if '{"labels":"dent"}' in region_label_list or '{"labels":"scratch"}' in region_label_list or '{"labels":"defect"}' in region_label_list:
					if np.unique(seg_mask).sum() < 1:
						#something wrong with segmentation mask , skip the image
						continue
					
					# create a head name , eg : both_, scratch_, dent_ etc
					name_head = ""
					name_head_w = ""
					
					if np.unique(dent_mask).sum() > 0 and np.unique(scratch_mask).sum() > 0:
						name_head = "both_"
					elif np.unique(dent_mask).sum() >0 :
						name_head = "dent_"
					elif np.unique(scratch_mask).sum() > 0:
						name_head = "scratch_"
					

					if np.unique(seg_mask_window).sum() > 0:
						name_head_w = "window_"


					print(name_head)


					# create new image name which is valid for dataset-code.py
					#eg : image = abc_blended.jpg 
					#     mask = abc_mask.png
					image_name_org_hood = name_head + image_name.replace(".jpg","_blended.jpg")
					image_name_seg_mask_hood = name_head + image_name.replace(".jpg","_mask.png")

					seg_mask_as_bool = copy.deepcopy(seg_mask)
					seg_mask_as_bool =  seg_mask_as_bool.astype(bool)

					#multiply image with segmentation binary mask

					#make a copy of the image
					copy_image_1 = copy.deepcopy(image)
					copy_image_2 = copy.deepcopy(image)
					copy_image_1 = copy_image_1 * seg_mask_as_bool


					if np.unique(seg_mask_window).sum() > 0:
						image_name_org_window = name_head_w + image_name.replace(".jpg","_blended.jpg")
						image_name_seg_mask_window = name_head_w + image_name.replace(".jpg","_mask.png")
						seg_mask_window_bool = copy.deepcopy(seg_mask_window)
						seg_mask_window_bool = seg_mask_window.astype(bool)
						copy_image_2 = copy_image_2 * seg_mask_window_bool

					#find contours for the seg-mask to get its bbox values,
					#then we use this values to crop the image
					
					contours1, hierarchy1 = cv2.findContours(seg_mask.copy(),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
					for i in contours1:
						all_x_1 = [ j[0][0] for j in i]
						all_y_1 = [ j[0][1] for j in i]
						xmin1,ymin1 = min(all_x_1), min(all_y_1)
						xmax1,ymax1 = max(all_x_1), max(all_y_1)
						break
					
					# create final mask ( channel==3), by stacking
					final_mask = np.dstack((dent_mask, scratch_mask, scratch_mask))
					final_mask = final_mask * seg_mask_as_bool
					# plt.imshow(final_mask)
					# plt.show()

					# crop both image and mask with bbox values 
					copy_image_1 = copy_image_1[ymin1:ymax1, xmin1:xmax1]
					final_mask = final_mask[ymin1:ymax1, xmin1:xmax1]

					#resize the image into 320
					copy_image_1= cv2.resize(copy_image_1, (320,320))
					final_mask = cv2.resize(final_mask, (320,320))
					# plt.imshow(image)
					# plt.show()
					# plt.imshow(final_mask)
					# plt.show()

					#similarly for the windhshield part we have to make it 
					contours2, hierarchy2 = cv2.findContours(seg_mask_window.copy(),  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

					for i in contours2:
						all_x_2 = [ j[0][0] for j in i]
						all_y_2 = [ j[0][1] for j in i]
						xmin2,ymin2 = min(all_x_2), min(all_y_2)
						xmax2,ymax2 = max(all_x_2), max(all_y_2)
						break
					
					# create final mask ( channel==3), by stacking
					#if defect label is used for window
					final_mask_window = np.dstack((defect_mask_window,defect_mask_window,defect_mask_window))
					final_seg_mask_window = np.dstack((seg_mask_window,bty_mask_window,bty_mask_window))
					#else
					#final_mask_window = np.dstack((dent_mask,dent_mask,dent_mask))
					if np.unique(seg_mask_window).sum() > 0:
						final_mask_window = final_mask_window * seg_mask_window_bool
						# plt.imshow(final_mask)
						# plt.show()
						# crop both image and mask with bbox values 
						copy_image_2 = copy_image_2[ymin2:ymax2, xmin2:xmax2]
						final_mask_window = final_mask_window[ymin2:ymax2, xmin2:xmax2]
						final_seg_mask_window = final_seg_mask_window[ymin2:ymax2,xmin2:xmax2]

						#resize the image into 320
						copy_image_2 = cv2.resize(copy_image_2, (320,320))
						final_mask_window = cv2.resize(final_mask_window, (320,320))
						final_seg_mask_window = cv2.resize(final_seg_mask_window,(320,320))
						# plt.imshow(image)
						# plt.show()
						# plt.imshow(final_mask)
						# plt.show()
						#write both image into same folder for the hood
						print(dest_defect_image+"door/"+image_name_org_hood)
						print(np.unique(copy_image_1))
						#exit("finish")

					if '{"labels":"dent"}' in region_label_list or '{"labels":"scratch"}' in region_label_list or '{"labels":"defect"}' in region_label_list:
						if os.path.isdir(dest_defect_image+"door"):
							cv2.imwrite(dest_defect_image+"door/"+image_name_org_hood,copy_image_1)
							cv2.imwrite(dest_defect_image+"door/"+image_name_seg_mask_hood,final_mask)
						else:
							os.makedirs(dest_defect_image+"door")
							cv2.imwrite(dest_defect_image+"door/"+image_name_org_hood,copy_image_1)
							cv2.imwrite(dest_defect_image+"door/"+image_name_seg_mask_hood,final_mask)
					#write both image into same folder for the windhshield part 
					if '{"labels":"dent"}' in region_label_list or '{"labels":"scratch"}' in region_label_list or '{"labels":"defect"}' in region_label_list:
						if np.unique(seg_mask_window.sum() > 0):
    							
							if os.path.isdir(dest_defect_image+"window"):
								cv2.imwrite(dest_defect_image+"window/"+image_name_org_window,copy_image_2)
								cv2.imwrite(dest_defect_image+"window/"+image_name_seg_mask_window,final_mask_window)
								cv2.imwrite(dest_defect_image+"window/"+image_name_org_window.split("_blended")[0]+"_seg_mask.png",final_seg_mask_window)	
							else:
								os.makedirs(dest_defect_image+"window/")
								cv2.imwrite(dest_defect_image+"window/"+image_name_org_window,copy_image_2)
								cv2.imwrite(dest_defect_image+"window/"+image_name_seg_mask_window,final_mask_window)
								cv2.imwrite(dest_defect_image+"window/"+image_name_org_window.split("_blended")[0]+"_seg_mask.png",final_seg_mask_window)	

				else:
						#here we are going to save non defect hood and all  and others there in csv only
						if (seg_mask.sum() > 0):
								final_mask_hood = np.dstack((seg_mask,bty_mask,bty_mask))
								#For saving the hood at the non defect images path
								if os.path.isdir(dest_non_defect_image+"door"):
									cv2.imwrite(dest_non_defect_image+"door/"+image_name,image)
									cv2.imwrite(dest_non_defect_image+"door/"+image_name.split(".jpg")[0] + "_mask.png",final_mask_hood)
								else:
									os.makedirs(dest_non_defect_image+"door")
									cv2.imwrite(dest_non_defect_image+"door/"+image_name,image)
									cv2.imwrite(dest_non_defect_image+"door/"+image_name.split(".jpg")[0] + "_mask.png",final_mask_hood)

						if (seg_mask_window.sum() > 0):
								final_mask_window = np.dstack((seg_mask_window,bty_mask_window,bty_mask_window))
								#For saving the window at non defect images path 
								if os.path.isdir(dest_non_defect_image+"window"):
									cv2.imwrite(dest_non_defect_image+"window/"+image_name,image)
									cv2.imwrite(dest_non_defect_image+"window/"+image_name.split(".jpg")[0] + "_mask.png",final_mask_window)
								else:
									os.makedirs(dest_non_defect_image+"window")
									cv2.imwrite(dest_non_defect_image+"window/"+image_name,image)
									cv2.imwrite(dest_non_defect_image+"window/"+image_name.split(".jpg")[0] + "_mask.png",final_mask_window)

				region_label_list = []
	#saving the dataframe for the other loc codes with new bbox and new label this is for the std thing 
	main_df = pd.DataFrame({"image_name":image_name_list,"xmin":xmin_list,"ymin":ymin_list,"xmax":xmax_list,"ymax":ymax_list,"label":label_list})
	main_df.to_csv(dest_non_defect_image+"/for_retraining_ssd.csv",index = False)


