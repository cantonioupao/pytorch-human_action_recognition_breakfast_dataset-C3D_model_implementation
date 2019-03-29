import os
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
import glob
import shutil


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'breakfast'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='breakfast', split='train', clip_len=16, preprocess=False): 
        self.root_dir, self.output_dir = Path.db_dir(dataset)  #this is defined from the path file python file dataset.py
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112
        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')#if the dataset directory exists then we continue to the next method

        if (not self.check_preprocess()) or preprocess:
            #if the output directory doesnt exist , then we have to go to the method and carry out preprocessing
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()    #after preprocessing is completed we move on

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):                #so for everything inside ucf101-->train
            for fname in os.listdir(os.path.join(folder, label)): #and for everything inside all the folders of ucf101--> train . So for example everything inside ucf101-->train-->ApplyMakeup
                self.fnames.append(os.path.join(folder, label, fname))   #add to the list that holds all the file names , the current file name . So at the end fnames will hold all the path like:   a) ucf101-->train-->ApplyMakeup-->ApplyMakeup123_g0   b)ucf101-->train-->Yolo-->Yolo_scn3_g0
                labels.append(label)     #add the next action label-activity label to the list that holds all the activities-action labels  . So it will hold ucf101-->train-->ApplyMakeup , ucf101-->train-->Yolo , ucf101-->train-->Typing, 

        assert len(labels) == len(self.fnames)  # so the length of the labels list needs to be always equal to the length of the fnames list . If this is false then the program halts
        print('Number of {} videos: {:d}'.format(split, len(self.fnames))) # so the number of train videos is equal to 2190 

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('dataloaders/actions_labels.txt'):
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'breakfast':
            if not os.path.exists('dataloaders/actions_breakfast_labels.txt'):
                with open('dataloaders/actions_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True


    def preprocess(self):#jumps to preprocess
        local_dir ="./break_not"     #local output_directory
        if not os.path.exists(local_dir):
               os.mkdir(local_dir)


        # Split train/val/test sets
        count_d = 3
        num_videos = 0
        num_actions = 0
        sframes,fframes,f_actions = [] , [] , []
        for file in os.listdir(self.root_dir):            #for any file inside the root directory 
            file_path = os.path.join(self.root_dir, file)  #file path will be the path of the joined filename
            for file2 in os.listdir(file_path): #create another loop to accommodate the second file
                file_path_angle = os.path.join(file_path, file2) #add another path to the folder 		
                #video_files=[name for name in os.list.dir(file_path_angle)] #this will give all the files on a list saved
                video_files = [f for f in glob.glob(os.path.join(file_path_angle,'*.avi'))]  #alternatively we can use the globe as mentioned
                for filename_actions in glob.glob(os.path.join(file_path_angle, '*.txt')):   #so for all the text files in etc.  breakfast->PO4->stereo
                    #Convert the text_file to a .avi extension to compare the names
                    compare_name = filename_actions.split('.')
                    #print(compare_name)
                    txt_to_avi="."+str(compare_name[1])+".avi"
                    index = video_files.index(txt_to_avi)#print(str(num_videos)+ "our video is"+str(video_files[num_actions])+"but the filename is"+str(filename_actions))
                    print("The textfile is   "+txt_to_avi+ "   whereas, the video file is   "+video_files[index])
                    with open(filename_actions, 'r') as f:
                    	lines = f.readlines()
                    for i,file_line in enumerate(lines):
                       space = file_line.rsplit(' ') #splits the frames with the action list
                       f_actions.append(file_line.split(' ')[1])  # actions list stores all the actions for each video
                       frames =space[0].split('-')  #frames get the initial and final frames for each action
                       sframes.append(frames[0])  #the sframes list stores all the start frames for each action
                       fframes.append(frames[1])  # the fframes list stores all the end frames for each action
                       #print("So for"+str(video_files[index])+"the text file was"+str(filename_actions[num_videos])+ "and the action was "+ str(f_actions[num_actions]))
                       self.process_video(file,file2,video_files[index],f_actions[num_actions],sframes[num_actions],fframes[num_actions],local_dir)
                       num_actions+=1
                num_videos+=1
        count_d+=1
        print('Preprocessing finished.')
        self.divide_dataset(local_dir)


    def divide_dataset(self, root_dir):
        if os.path.exists(self.output_dir):
            if not os.path.exists(os.path.join(self.output_dir,'train')):
                os.mkdir(os.path.join(self.output_dir,'train'))  #create the first directory
                os.mkdir(os.path.join(self.output_dir,'val')) # 2nd directory
                os.mkdir(os.path.join(self.output_dir,'test')) #3 directory
        else:
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir,'train')) #create the first directory
            os.mkdir(os.path.join(self.output_dir, 'val')) # 2nd directory
            os.mkdir(os.path.join(self.output_dir, 'test')) #3 directory
        # Split train/val/test sets
        for file in os.listdir(root_dir):            #for any file inside the root directory 
            action_folder_path = os.path.join(root_dir, file)  #file path will be the path of the joined filename
            frame_folder_files = [name for name in os.listdir(action_folder_path)]  
            train_and_valid, test = train_test_split(frame_folder_files, test_size=0.2, random_state=42)  #this signifies that our test dataset will e the 20% of the dataset - sklearn function#
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)  #this signifies that the validation dataset will be 20% of it , leaving 60% for training #
        
            #Define the training, validation and testing directories that the frame folders will be moved to.
            train_dir = os.path.join(self.output_dir, 'train',file) #creates the path for break->train->stir_milk->PO3_stereo_cereals_115-322 
            val_dir = os.path.join(self.output_dir, 'val', file) #creates the path for break->train->stir_milk 
            test_dir = os.path.join(self.output_dir, 'test',file) #creates the path for break->train->stir_milk
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
    
            for frame_folders in train:
                #get only the last directory of the path frame_folders
                frame_folder = os.path.join(root_dir,file,frame_folders)
                shutil.move(frame_folder,train_dir)
            for frame_folders in val:
                frame_folder = os.path.join(root_dir,file,frame_folders)
                shutil.move(frame_folder,val_dir)
            for frame_folders in test:
                frame_folder = os.path.join(root_dir,file,frame_folders)
                shutil.move(frame_folder,test_dir)
        print('Dataset Division finished.')        


    def process_video(self,file,file2,video,f_actions,sframes,fframes,save_dir):  #to explain:f_actions holds the actions of the video file ,  sframes is the list that holds all the starting frames ,  fframes is the list that holds all the final -finishing frames
        # Initialize a VideoCapture object to read video data into a numpy array
        head, tail = os.path.split(video)
        video_filename = tail.split('.')[0]   #from the video file we take only the name of it and set it as the name of the new folder created (for us this will not be needed)
        #print(video_filename)
        capture = cv2.VideoCapture(video)  # now using the cv2 library we stat a capture for the video in the root directory path


        #convert the string list to an integer list 
        #sframes=list(map(int,sframes))
        #fframes=list(map(int,fframes))

        
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  #get numbers of frames for the video
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  #get the frame width
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) #get the frame height

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0 #it starts from 0 
        i = 1 #it starts at 1 because all are frames start from 1 not 0
        retaining = True
        #stop = 0  #initialisation of stop 
        #stop_count = 0 #set it as 0 initially to get the first element of the sframes and fframes lists
        #three=3

        #stop = "".join(map(str, fframes)) #define stop as the final frame etc.110
        #start ="".join(map(str, sframes))
        #print(stop)
        #print(start)
        #convert to int
        stop = int(fframes)
        start =int(sframes)
        f_name = f_actions   #this will be the action name 
        dataset_directory =save_dir 
        #print(frame_count )#count, frame_count, retaining)
        while (count < frame_count and retaining):  #so as long as the counter is less than the total number of frames and retaining variable is true
            retaining, frame = capture.read()   # we keep reading the frames from the video
            #print("The number of "+ str(count) +"and the number of frames "+ str(stop))
            if frame is None:
                continue
            #dataset_directory = os.path.dirname(save_dir) #remove the previous folder-component of the path ,we are left only with the directory. etc. C:/break/train
            save_dir1 = os.path.join(dataset_directory,str(f_name)) #this is the final save directory. It will be of the form C:/break/train/stir_milk
            if not os.path.exists(os.path.join(save_dir1)): #create the path of the new folder with the video name if it doesnt exist
                os.mkdir(save_dir1)

            video_filename_path = str(file)+"_"+str(file2)+"_"+str(video_filename)+"_"+str(start)+"_"+str(stop)
            if not os.path.exists(os.path.join(save_dir1, video_filename_path)): #create the path of the new folder with the video name if it doesnt exist
                os.mkdir(os.path.join(save_dir1, video_filename_path)) #make it
            #print(count % EXTRACT_FREQUENCY)
            if count>=start: #count % EXTRACT_FREQUENCY == 0:
                #define the video_filename path
                #video_filename_path = str(file)+"_"+str(file2)+"_"+str(video_filename)+"_"+str(sframes[stop_count])+"_"+str(fframes[stop_count])
                #print("write")   #store all the frames accordning to its frame number .The dirname removes the last folder-component from the save_directory
                if(count==stop):
                    #print( "the" + str(stop)+ " is " + str(video_filename_path)+"at frame"+str(count)) #check which video stops
                    break
                    #if statement to check if we need to switch the frame packet
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir1, video_filename_path, '0{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1  #and keep counting until the video is fully read and saved
        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        while frame_count <= 16:
        	frame_count += 1
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        i = 0
        frame = None
        for _, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
            i += 1
        #if frame is None:
        	#print(file_dir)
        while i < 16:
        	buffer[i] = frame
        	i+=1

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='breakfast', split='test', clip_len=8, preprocess=False)  #execute the class for the test dataset ( i have the impression the validation dataset is never executed)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
              break
