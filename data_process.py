import os
import csv
import numpy as np
from PIL import Image
from keras.preprocessing import image
import sklearn
from random import shuffle

def csv_reader(csv_file, skipHeader = True):
    
      lines = []
      with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            if skipHeader: #To skip the header as the name implies..
                  next(reader, None)      
            
            for line in reader:
                  lines.append(line)
      return lines

def images_generator(csv_lines, img_path, batch_size=32):

      num_samples = len(csv_lines)
      while 1: # Loop forever so the generator never terminates
            shuffle(csv_lines)
            for offset in range(0, num_samples, batch_size):
                  
                  batch_samples = csv_lines[offset:offset+batch_size]
                  
                  car_images = []
                  steering_angles = []

                  for batch_sample in batch_samples:

                        # read in images from center, left and right cameras
                        img_center = image.load_img((img_path + batch_sample[0].strip()), target_size=(160, 320))
                        img_center = image.img_to_array(img_center)

                        img_left = image.load_img((img_path + batch_sample[1].strip()), target_size=(160, 320))
                        img_left = image.img_to_array(img_left)
                  
                        img_right = image.load_img((img_path + batch_sample[2].strip()), target_size=(160, 320))
                        img_right = image.img_to_array(img_right)

                        steering_center = float(batch_sample[3])

                        # create adjusted steering measurements for the side camera images
                        correction = 0.5 # this is a parameter to tune
                        steering_left = steering_center + correction
                        steering_right = steering_center - correction

                        # add images and angles to data set
                        car_images.extend([img_center,img_left,img_right])

                        steering_angles.extend([steering_center,steering_left,steering_right])


                        #Flipping the same image and adding it to the dataset
                        img_center_flipped = np.fliplr(img_center)
                        img_left_flipped = np.fliplr(img_left)
                        img_right_flipped = np.fliplr(img_right)

                        steering_center_flipped = steering_center* (-1.0)
                        steering_left_flipped = steering_left * (-1.0)
                        steering_right_flipped = steering_right * (-1.0)

                        # add images and angles to data set
                        car_images.extend([img_center_flipped, img_left_flipped, img_right_flipped]) 

                        steering_angles.extend([steering_center_flipped, steering_left_flipped, steering_right_flipped])


                  X = np.array(car_images)
                  y = np.array(steering_angles)

                  yield  sklearn.utils.shuffle(X, y)