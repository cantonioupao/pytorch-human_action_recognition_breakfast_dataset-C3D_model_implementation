import os
import shutil
import torch
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

output_dir ="./break"
if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'train'))  #create the first directory
        os.mkdir(os.path.join(output_dir, 'val')) # 2nd directory
        os.mkdir(os.path.join(output_dir, 'test')) #3 directory

        # Split train/val/test sets
        for file in os.listdir(output_dir):            #for any file inside the root directory 
            action_folder_path = os.path.join(output_dir, file)  #file path will be the path of the joined filename
            for file2 in os.listdir(action_folder_path): #create another loop to accommodate the second file
                video_folder_path = os.path.join(action_folder_path, file2) #add another path to the folder 	
                frame_folder_files = [name for name in os.listdir(video_folder_path)]	
                train_and_valid, test = train_test_split(frame_folder_files, test_size=0.2, random_state=42)  #this signifies that our test dataset will e the 20% of the dataset - sklearn function#
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)  #this signifies that the validation dataset will be 20% of it , leaving 60% for training #
                
                #Define the training, validation and testing directories that the frame folders will be moved to.
                train_dir = os.path.join(output_dir, 'train',file) #creates the path for break->train->stir_milk->PO3_stereo_cereals_115-322 
                val_dir = os.path.join(output_dir, 'val', file) #creates the path for break->train->stir_milk 
                test_dir = os.path.join(output_dir, 'test',file) #creates the path for break->train->stir_milk
                print(str(file) + "------>"+str(file2)+"------>"+str(train_dir))
                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                if not os.path.exists(val_dir):
                    os.mkdir(val_dir)
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)

                for frame_folders in train:
                    print(frame_folders)
                    #get only the last directory of the path frame_folders
                    frame_folder = os.path.basename(os.path.dirname(frame_folders))
                    print(frame_folder)
                    shutil.move(frame_folders, os.path.join(train_dir,frame_folder))

                for frame_folders in val:
                    frame_folder = os.path.basename(os.path.dirname(frame_folder))
                    shutil.move(frame_folders,frame_folder)

                for frame_folders in test:
                    frame_folder = os.path.basename(os.path.dirname(frame_folder))
                	shutil.move(frame_folders, frame_folder)      
        print('Dataset Division finished.')        
