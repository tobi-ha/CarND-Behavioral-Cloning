# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes (model.py lines 70-74) 

The model includes RELU layers to introduce nonlinearity (code lines 70-74), and the data is normalized in the model using a Keras lambda layer (code line 69). 

#### 2. Attempts to reduce overfitting in the model

The model contains subsampling in order to reduce overfitting (model.py lines 70-74). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving counter-clockwise. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with augmenting my training data by using all three images (center, left & right) and flipping all of the images. For the left and right images, I added/subtracted a correction steering angle of 0.3. 

Flipping the images should help the model zu generalize better. 

I also edited the image data by cropping 70 pixels off the top and 20 pixels off the bottom. 

Before feeding the data into the model, I normalized the data with a Keras Lambda layer. Therefore I devided the image data by 255 and subtracted 0.5.

For the behavioural cloning project, I decided to use the Nvidia model architecture: 
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

That model includes 5 convolutional layers and 5 fully connected layers. 
The non-linearity is created by relu-activations. 

30 % of the data were used for validation. 

Training 3 epochs created only 0,0241 validation loss and turned out to drive very smooth.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Thereby I only used the control via mouse because that made it possible to achieve a smooth driving behaviour. After the smooth center lane driving, I recorded a few scenes in which I brought the vehicle back to the center lane by recovering from the left and right of the track. 

With these data, the model trained smooth driving and also recovering from the sides. 

For training, I shuffled the data used an adam optimizer so that manually tuning the learning rate wasn't necessary.