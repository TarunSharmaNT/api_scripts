##########################Edited March 26 Tarun Sharma######################################
###############################################################################


#importing the libraries 
import pandas as pd 
import cv2 
import glob 
import shutil 
import matplotlib.pyplot as plt 

### params 
image_source_folder = "/home/tarun/Number_Theory/New_Data_Preparation/Data/api_data_march_15/"
std_api_csv_path ="/home/tarun/Number_Theory/New_Data_Preparation/api_test/std_api_test/test_folder_dest/main.csv"
defect_api_csv_path = "/home/tarun/Number_Theory/New_Data_Preparation/api_test/defect_api_test/csv_data/2021_03_181616057351defect_api_result.csv"
destination_folder = "/home/tarun/Number_Theory/New_Data_Preparation/Data/dest_images_new/"
#############################################

df_defect = pd.read_csv(defect_api_csv_path)
df_std = pd.read_csv(std_api_csv_path)

#############################################

for img in glob.glob(image_source_folder + "*.jpg"):

	image_name = img.split('/')[-1]

	if image_name in df_std['image_name'].tolist():
		
		image = cv2.imread(img)

		sub = df_std.loc[df_std['image_name']==image_name,]

		for i in range(len(sub)):

			data = sub.iloc[i,]
			
			location_name = data['pred_loc_code']
			if location_name == "dummy" :
    			#if it is dummy in std then no need to check further simply copy the image 
				shutil.copy(img, destination_folder)
				continue


			xmin = int(data['p_xmin'])
			ymin = int(data['p_ymin'])
			xmax = int(data['p_xmax'])
			ymax = int(data['p_ymax'])

			image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0),3)
			image = cv2.putText(image, location_name ,(xmin + 20, ymin + 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)


		if image_name in df_defect['image_name'].tolist():

			sub = df_defect.loc[df_defect['image_name']==image_name,]

			for i in range(len(sub)):

				data = sub.iloc[i,]

				status = data['detection_status']

				if status != "True":
					continue
				else :
					xmin = int(data['xmin'])
					ymin = int(data['ymin'])
					xmax = int(data['xmax'])
					ymax = int(data['ymax'])

					image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0),3)
		else:
			pass
		cv2.imwrite(destination_folder+image_name, image)


	else:
		# simply copy the image nothing to draw 
		continue
		shutil.copy(img, destination_folder)




