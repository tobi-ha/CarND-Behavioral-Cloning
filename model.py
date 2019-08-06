import csv
import cv2
import numpy as np
#import matplotlib.pyplot as plt

data_dir = '/opt/data/'

### read the CSV-file "driving_log.csv" into lines
lines = []
with open(data_dir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

### read the steering angle (-> y_train) and images (-> X_train)
images = []
measurements = []
for line in lines:
    image_path = data_dir+'IMG/'
    ### paths of the images (center, left, right)
    path_center = image_path + line[0].split('/')[-1]
    path_left = image_path + line[1].split('/')[-1]
    path_right = image_path + line[2].split('/')[-1]
    
    ### load the images (center, left, right)
    image_center = cv2.imread(path_center)
    image_left = cv2.imread(path_left)
    image_right = cv2.imread(path_right)
    
    ### steering angles (center, left, right)
    correction = 0.3 # this is a parameter to tune
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    ### add the images and steering angles to the list
    images.append(image_center)
    images.append(image_left)
    images.append(image_right)
    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)
    
    ### also add the l-r-flipped images and steering angles to the list
    images.append(np.fliplr(image_center))
    images.append(np.fliplr(image_left))
    images.append(np.fliplr(image_right))
    measurements.append(steering_center*(-1)) 
    measurements.append(steering_left*(-1))
    measurements.append(steering_right*(-1))

# convert X_train and y_train into numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

print("X_train.shape = ", X_train.shape)
print("y_train.shape = ", y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

### NETWORK DESIGN (NVIDIA MODEL)
model = Sequential()
### cut 50 pix off the top and 20 pix from the bottom of the images
model.add(Cropping2D(cropping=((70,20),(0,0)), input_shape=(160,320,3)))
### normalize and go for zero mean
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(90,320,3)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=3)

model.save('model.h5')