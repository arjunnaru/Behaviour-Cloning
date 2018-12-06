
# coding: utf-8

# ## Model
# 
# This was the file used in the AWS Instance to train the model.

# In[1]:


import csv
import cv2
import numpy as np


# ## Training Data
# Data was collected vy driving the car in the in the simulator provided. Multiple data sets were collected and used. The two that gave the best result have been shown in teh code below. All the data sets that were collected are attched in the submission. A brief Description of the data collection techniques and cahllenges faced is given below.
# 
# ### Data collection
# 
# The different data files that were collected are as follows:
# 1. Anticlockwise on Track - Fast
# 2. Clockwise on Track - Slow
# 
# The data sets were collected at different speed slow as well as fast to ensure the vehicle could react to different situations. All the data sets that were finally used are atleast 1 lap long. a combination of the two above was good enough to tbe able to run the car in Autonomous mode and meet the requirements. 
# 
# A Oython generator was not needed as the dat generated was able to run in the memory providede without any problems. Thus a generator was not implemented for this part of the project
# 
# ### Data Augmentation
# Now the data was augmented using two methods
# 
# 1. Using the left and right Camera Images to exaggerate a recovery Action
# 2. Using mirror images of all the data 
# 
# #### Using Left and right Images
# The data collected was based on 3 cameras all facing forward but mounted on three different locations on the car Hood. 
# 
# The car is driven using only the center camera. Thus I made teh assumption taht if the center camera saw what the left camera saw then we are veering to the left and thus need to make a motion to the right to come back to the center of the lane. 
# 
# Similarly if the image seen in teh right camera is seen in teh center camera then the car is too far right nad should veer left. Thus the data was cahnged to account for this. This tripled the number of data points we have. 
# 
# Thus correction added
# 
# 1. Right image = Turn left by 0.2 Deg (Arbitrary Value)
# 2. Left Image = turn right by 0.2 Deg
# 3. Center Image - No Correction
# 
# 
# #### Using Mirror Images
# The track is a contant set of left turns (Anticlockwise) and rigth turns (Clockwise). Thus when we drive on this track we are making the model biased towards one side of the turn. Yes driving both ways means we teach teh car to turn both left and right. But while driving Anticlockwise considering the array of images we only teach it to turn left. 
# 
# Thus I mirrored all the data images adn reversed the steering angles for them to also teach the car to turn in the other direction. This again doubled the dataset. 
# 
# Thus the dataset was multiplied by 6 to teach the car different features required.
# 
# 
# 

# In[9]:


# Creating empty arrays to store the images and the Measurements
images = []
measurements = []

#AntiClockwise data 3 (Accidentally named it CW3 but data was actually ACW3)
lines = []
#Open Data
with open('/home/carnd/CW3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Sort through all lines - Extract path from file - Point to path in instance - Add the image to the array
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '/home/carnd/CW3/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        
        #Depending on if the image is the left image or the right image or the center image add a correction. 
        # The correction is as follows:
            #Right - Move left by 0.2 Deg
            #Center - No Correction
            #Left - Move Right by 0.2 Deg
            
        if i ==0:
            c = 0
        elif i == 1:
            c = 0.2
        else:
            c = -0.2

        measurement = float(line[3])
        measurement = measurement+c
        measurements.append(measurement)

print(np.array(images).shape)
print(np.array(measurements).shape)

# Do the same to the next data set
lines = []
with open('/home/carnd/CW1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '/home/carnd/CW1/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        
        if i ==0:
            c = 0
        elif i == 1:
            c = 0.2
        else:
            c = -0.2

        measurement = float(line[3])
        measurement = measurement+c
        measurements.append(measurement)

print(np.array(images).shape)
print(np.array(measurements).shape)

#Create Augmented Image and measurement (Steering Angle) Array 
augmented_images, augmented_measurements = [],[]
for image,measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

print(np.array(augmented_images).shape)
print(np.array(augmented_measurements).shape)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# ## Model Architecture Used
# 
# A convolution Neurral network with a flatten nural network with one output node was used to train the model. 
# 
# The data was first Noralized adn mean centered so taht any features that were biased in teh data do not affect the learning too much.
# 
# The image was then cropped to only look at the road and remove the part of the image that looks at the trees and the sky. 
# 
# The CNN is made up of 4 Convolution Layers with Relu Activation and Max Pooling at every stage. 
# 
# The filter sizze for the first two layers is 5X5 and the filter size for the last two layers is 3X3
# 
# The depth begins at 3 for the image and then is increased to 6 then 18 then 36 and finally 48 by the final layer. 
# 
# A dropout layer was added after the final convolution layer to ensure we do not overfit tot he data. I had initially added dropouts to every layer and was running into significant issues with learning the features and thus kept decreasing the number of dropout layers until this configuration gave me the desired result.
# 
# After this we have a flat Neural network going from 512 - 256 - 128 - 32 - 1 nodes. Th eoutput of this is the steering angle of the car.
# 

# In[10]:


# Define Model used to train - model Architecture is explained in the next Block
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# In[11]:


model = Sequential()
#Data is Normalized and Mean centered
model.add(Lambda(lambda x:((x/255.0)-0.5),input_shape = (160,320,3)))
#Data is cropped such that a part of the sky that is not required is removed
model.add(Cropping2D(cropping=((70,25),(0,0))))
#Convolution layer with filter of 5X5 and depth of 6 with a relu activation
model.add(Convolution2D(6,5,5,activation='relu'))
#Max Pooling along a 2X2 filter
model.add(MaxPooling2D((2,2)))
#Convolution layer with filter of 5X5 and depth of 18 with a relu activation
model.add(Convolution2D(18,5,5,activation='relu'))
#Max Pooling along a 2X2 filter
model.add(MaxPooling2D((2,2)))
#Convolution layer with filter of 3X3 and depth of 36 with a relu activation
model.add(Convolution2D(36,3,3,activation='relu'))
#Max Pooling along a 2X2 filter
model.add(MaxPooling2D((2,2)))
#Convolution layer with filter of 3X3 and depth of 4 with a relu activation
model.add(Convolution2D(48,3,3,activation='relu'))
#Max Pooling along a 2X2 filter
model.add(MaxPooling2D((2,2)))
#Dropout layer to reduce overfitting - ignoring half th generated points
model.add(Dropout(0.5))
# Flatten the network
model.add(Flatten())
#Layer 2 512 Nodes with relu activation
model.add(Dense(512))
model.add(Activation('relu'))
#Layer 3 256 Nodes with relu activation
model.add(Dense(256))
model.add(Activation('relu'))
#Layer 4 128 Nodes with relu activation
model.add(Dense(128))
model.add(Activation('relu'))
#Layer 5 32 Nodes with relu activation
model.add(Dense(32))
model.add(Activation('relu'))
#Output layer with a steering angle prediction
model.add(Dense(1))


# ## Trining the model on the data
# 
# The data is split into a training set and a validation set. 
# 
# An adam optimiser is then used to train the model and get the necessary output.
# 
# The model was then saved and used to drive the Simulator on the local machine. 

# In[12]:


#adam Optimiser used so Learning rate was automatically tuned
model.compile(loss ='mse',optimizer = 'adam')
#Data was split into Test and Validation Set and then fit onto the model described above
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=2)
#Model is saved
model.save('model.h5')

