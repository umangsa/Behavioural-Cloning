#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/final_distribution.png "Data distribution for training"
[image2]: ./examples/initial_distribution.png "Initial distribution of training data"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I tried multiple model architectures to solve this project problem. I initially started with building a simple network using a convolutional layer followed by ReLU. Based on the results I was getting, I added layers. However, my model quickly became very big with more than 2M parameters, without getting me the results I was looking for.

I then implemented the NVidia model. With this model, I was able to drive the car for a certain distance. However, I was unable to make the car take the sharp curves after the bridge. 

I finally switched to the CommaAI model for steering control. I made a few changes to the model
* I cropped the the top 65 pixels and bottom 20 pixels to remove details that would only make the network larger or complex
* I resized the resulting image to 66x200. 
* Converted the color space to HSV. On the forum, many people suggested YUV would be a better color space to use.
* Normalize and zero center the input images

My network has 3 pairs of Convolution, Activation layers. I used ELU as the activation layer. THis is followed by Flatten, Dropout to reduce over fitting, and Dense layers.

Following is the my model architecture

Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 75, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
lambda_3 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 15, 49, 16)        3088      
_________________________________________________________________
elu_1 (ELU)                  (None, 15, 49, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 6, 23, 32)         12832     
_________________________________________________________________
elu_2 (ELU)                  (None, 6, 23, 32)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 1, 10, 64)         51264     
_________________________________________________________________
flatten_1 (Flatten)          (None, 640)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 640)               0         
_________________________________________________________________
elu_3 (ELU)                  (None, 640)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               328192    
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
elu_4 (ELU)                  (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 513       
=================================================================
Total params: 395,889
Trainable params: 395,889
Non-trainable params: 0


This network has fewer paramters and hence quicker to train. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 173 and 176). I also attempted to add L2 regularization. However, I found that the final network after training was not able to drive well. I ended up removing the L2 regularization and just used the dropout layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 209-219). I added data augmentation so that the network would see a wide range of different images during training.

Following techniques were used for data augmentation:
* Randomly chose either of the left, center or right camera
* Randomly flipped the image left-right
* Random translation of the images on the X Axis. Added steering angle compensation for this translation. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 215). Initially I was setting the Learning Rate of the Adam optimizer to 1e-5. I found that the validation error progressively dropped for the 50 epochs. However, the car was still not driving very smoothly. There were a few areas like the dirt track after the bridge where the car made sudden changes and would drive on the shoulder for a few frames. 

When I removed the LR=1e-5 initialization, I saw that the validation error no longer made a smooth lower progression. After a few epochs, the validation error started making wild swings, indicating that my Learning Rate may have been too high. However, the resulting network weights made the car drive much better, trying to remain in the center for most times. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I have already covered many of my architectural choices in sections above. I shall focus on other parts of my process of achieving the goal.

When I first started, the car would immediately drive off the track and race towards the water. I was using just the center camera images without any changes. 

I then flipped the images and was successful in driving on the track, but going circles.

I added the images from the Left and right cameras and flipped them too. I was able to get the car to go some distance on the track. But would hit the walls of the bridge.

Based on the suggestions in the tutorial videos, I started collecting a lot of training data. I had over 40K images. But it only made my model worse. Soon I got to the point where the car would no longer stay on the road. It was a major setback for me.

I decided to start all over again and just focus on the Udacity data and add images only for the problem areas. However, I was not able to solve the problem with this either.

My final dataset used only the Udacity data set, filtering of data and augmenting data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 160-179). This has been inspired from the CommaAI network. I found this network was very compact and fast to train. 

####3. Creation of the Training Set & Training Process

The initial training data that came with the project looked like this

![alt text][image2]

There were a lot of samples around steering angle zero and very little data for larger angles. As I recorded more data, this distribution got skewed further. I kept adding lot of samples with steering angle 0. This was causing my model to overfit towards zero angle and was root cause to lots of my problems. 

I started tweaking the data so that I got a zero centered, but near uniform distribution of data. This would allow my network to see a larger variety of angles, and hence generalize better. I ignored samples that were taken at very low speeds as well as 75% of the samples that were near 0 steering angle. Higher steering angles that had very few samples were added multiple times. 

My final input data distribution, before augmentation looked like:

![alt text][image1]

To augment the data sat, I also flipped images and angles. I also did random translations of the images on the X axis and added steering compensation. 

After the collection process, I had 50K number of data points. I split this 80-20 for training-validation. I then preprocessed this data by cropping top 65 and bottom 25 pixels, resize image to 66x200 (although aspect ratio was not maintained), converted the HSV color space and normalized the images


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting, although I am not sure that the validation error was giving me meaningful pointer to my final result. I tested weights from several epochs and finally chose the weights from the 36th epoch as ideal.
