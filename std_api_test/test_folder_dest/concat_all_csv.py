#################Author Tarun Sharma ######################
#################Dated March 5#############################
import pandas as pd
import numpy as np
import glob

main_df = pd.DataFrame()

Parent_folder = "/home/tarun/Number_Theory/New_Data_Preparation/api_test/std_api_test/test_folder_dest/"

for file in glob.glob(Parent_folder+"*.csv"):
	main_df = pd.concat([main_df,pd.read_csv(file)],axis=0)
	print(main_df.shape)	

#print(main_df.shape)

#exit("finish")

main_df.to_csv("main.csv",index = False)

exit("finish")

