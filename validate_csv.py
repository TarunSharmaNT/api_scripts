###########################   Author Tarun Sharma ##############################################
###########################   Dated March 18 ###################################################
########################### I can validate whenever i get the csv ##############################

#importing the libraries 
import pandas as pd
import numpy as np
import glob
import shutil
import cv2
import matplotlib.pyplot as plt


#image Folder 
#parent_folder = "/home/tarun/Number_Theory/New_Data_Preparation/Data/api_data_march_15/"
parent_folder ="/home/tarun/Number_Theory/New_Data_Preparation/Validation_csv/user_5/"
df = pd.read_csv("/home/tarun/Number_Theory/New_Data_Preparation/Rear_door/Varsha/via_export_csv (3).csv")

print(df.shape)

image_name_list = df["filename"].unique().tolist()
#iterating over the image 
for img in glob.glob(parent_folder + "*.jpg"):
    image_name = img.split("/")[-1]

    if image_name not in image_name_list:
        continue

    image = cv2.imread(img)
    sub = df.loc[df["filename"]==image_name,]

    for i in range(len(sub)):
        
        sub_data = sub.iloc[i,]

        loc = sub_data["region_attributes"]

        print(loc)
        if loc in ['{"labels":"dent"}','{"labels":"door"}','{"labels":"scratch"}','{"labels":"defect"}','{"labels":"window"}']:
            #plt.imshow(image)
            #plt.savefig("aniket/labelled/"+image_name)
            #plt.show()
            #continue
            #exit("first debug point")
            coordinates = sub_data["region_shape_attributes"]
            coordinates = eval(coordinates)



            #handling the above exception 
            print(coordinates)
            
            list_of_keys = [*coordinates.keys()]
            print(list_of_keys) 

            if 'x' in list_of_keys or 'y' in list_of_keys or 'height' in list_of_keys or 'width' in list_of_keys:
                continue

            #print(getList(coordinates.keys()))
            
            #exit("first debug")

            z = list(zip(coordinates['all_points_x'], coordinates['all_points_y']))
            contours = np.array([list(x) for x in z])

            #where seg_mask and the bty_mask is the single channel image             
            image = cv2.fillPoly( image, np.int32([contours]) , (2550,0,0))
            #bty_mask = cv2.polylines(image, np.int32([contours]),  True, (255), 3) 

        elif loc in '{"labels":"HOOD"}' or loc in '{"labels":"WINDSHIELD"}' or loc in '{"labels":"FRONT_VIEW"}' or loc in '{"labels":"REAR_VIEW"}' or loc in '{"labels":"REAR_45_LEFT"}' or loc in '{"labels":"FRONT_45_LEFT"}' or loc in '{"labels":"FRONT_45_RIGHT"}' or loc in '{"labels":"REAR_45_RIGHT"}' or loc in '{"labels":"DRIVER_SIDE"}' or loc in '{"labels":"PASSENGER_SIDE"}' or loc in '{"labels":"STEERING_WHEEL"}' or loc in '{"labels":"GEARSHIFT"}' or loc in '{"labels":"AIR_INTAKE"}' or loc in '{"labels":"DASHBOARD"}' or loc in '{"labels":"HEAD_LIGHT_RIGHT"}' or loc in '{"labels":"HEAD_LIGHT_LEFT"}' or loc in '{"labels":"FRONT_WINDOW_PSIDE"}' or loc in '{"labels":"DSIDE_FRONT_DOOR"}' or loc in '{"labels":"DSIDE_REAR_DOOR"}' or loc in '{"labels":"REAR_WINDOW_DSIDE"}' or loc in '{"labels":"PSIDE_FRONT_DOOR"}' or loc in '{"labels":"PSIDE_REAR_DOOR"}' or loc in '{"labels":"FRONT_WINDOW_DSIDE"}' or loc in '{"labels":"REAR_WINDOW_PSIDE"}':
            coordinates = sub_data["region_shape_attributes"]
            coordinates = eval(coordinates)

            list_of_keys = [*coordinates.keys()]
            print(list_of_keys)
            if 'all_points_x' in list_of_keys or 'all_points_y' in list_of_keys:
                continue
            
            xmin = int(coordinates["x"])
            ymin = int(coordinates["y"])
            xmax = int(coordinates["width"])+xmin
            ymax = int(coordinates["height"])+ymin
            image = cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),3)
            #pass
        elif loc in '{}' or loc in '{undefined}':
            continue
        
        plt.imshow(image)
        #plt.show()
        plt.savefig("Varsha/labelled1/"+image_name)
        plt.close()


##############################End of the Code #################################################################
###############################################################################################################
    




