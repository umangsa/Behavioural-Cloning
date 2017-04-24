import csv
import os, os.path
import cv2
import numpy as np
import tensorflow as tf

# import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import ELU, Lambda, Cropping2D
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend.tensorflow_backend as K

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS
BATCH_SIZE = 32
# steering_offset = 0.1595

# Offset to add when using left or right camera
steering_offset = 0.25

# Offset correction to be made dugin camera translation. 
# This is the compensation per pixel
translation_offset = 0.02




# Perform image translation as part of augmentation
def image_translation(image, angle):
	# Compute X translation
	x_translation = 100 * (np.random.rand() - 0.5)
	angle += x_translation * translation_offset
	
	# Form the translation matrix
	translation_matrix = np.float32([[1, 0, x_translation], [0, 1, 0]])
	# Translate the image
	image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
	return image, angle


# Data augmentaion 
def random_modify(image, angle):
	
	image, angle = image_translation(image, angle)
	return image, angle

# Image generator
def generator(samples, batch_size=32, training=True):
	num_samples = len(samples)
	lines = []

	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			measurements = []

			for line in batch_samples:

				# Randomly choose between camera. 
				cam = np.random.randint(1,3)
				if cam == 1:
					# left camera
					img_path = os.path.join("../data", line[1].strip())
					measurement = float(line[3]) + steering_offset
				elif cam == 2:
					# center camera
					img_path = os.path.join("../data", line[0].strip())
					measurement = float(line[3]) 
				elif cam == 3:
					# right camera
					img_path = os.path.join("../data", line[2].strip())
					measurement = float(line[3]) - steering_offset

				if os.path.isfile(img_path):
					image = cv2.imread(img_path)

					# Convert to RGB. opencv uses BGR, while we will get 
					# RGB from the simulator
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					
					# Randomly flip images
					flip_prob = np.random.uniform()
					if flip_prob < 0.5 :
						image = cv2.flip(image, 1)
						measurement = -measurement

					# Random augmentation of data
					if training:
						image, measurement = random_modify(image, measurement)

					image = np.array(image)
					images.append(image)
					measurements.append(measurement)
				else:
					print("Missing file {}".format(img_path))

			X = np.array(images)
			y = np.array(measurements)
			yield shuffle(X, y)

def nvidia_model():
	input_shape=(160, 320, 3)
	model = Sequential()
	model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=input_shape))
	model.add(Lambda(lambda image: K.tf.image.resize_images(image, (33, 100))))
	# model.add(Lambda(lambda image: K.tf.image.resize_images(image, (66, 200))))
	model.add(Lambda(lambda image: K.tf.image.rgb_to_hsv(image)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))
	model.add(Conv2D(24, kernel_size=5, strides=(2, 2), padding="same"))
	model.add(ELU())
	model.add(Conv2D(36, kernel_size=5, strides=(2, 2), padding="same"))
	model.add(ELU())
	model.add(Conv2D(48, kernel_size=5, strides=(2, 2), padding="same"))
	model.add(ELU())
	model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding="same"))
	model.add(ELU())
	model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding="same"))
	model.add(ELU())
	model.add(Dropout(.5))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(ELU())
	model.add(Dense(50))
	model.add(ELU())
	model.add(Dense(10))
	model.add(ELU())
	model.add(Dense(1))
	return model

# modified version of comma ai model for steering control
# working at 20 MPH with udacity reference data upto the bridge
def commaai_model():
	input_shape=(160, 320, 3)
	model = Sequential()

	# Following operations are performed on training as well as runtime images
	model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=input_shape))
	model.add(Lambda(lambda image: K.tf.image.resize_images(image, (66, 200))))
	model.add(Lambda(lambda image: K.tf.image.rgb_to_hsv(image)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))

	model.add(Conv2D(16, kernel_size=8, strides=(4, 4), padding="valid"))
	model.add(ELU())

	model.add(Conv2D(32, kernel_size=5, strides=(2, 2), padding="valid"))
	model.add(ELU())

	model.add(Conv2D(64, kernel_size=5, strides=(2, 2), padding="valid"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())

	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())

	model.add(Dense(1))
	return model

samples = []
with open('../data/udacity_driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:

		# ignore images that have very low speed
		if float(line[6].strip()) > 10:
			measurement = float(line[3].strip())

			# most of the data is aroung 0 degree steering
			# take only 25% of these images to get a 
			# balanced dataset
			if measurement >= 0.0 and measurement <= 0.08:
				prob = np.random.random()
				if prob >= 0.75:		
					samples.append(line)

			# These angles needs more data, add the images multiple times
			elif abs(measurement) > 0.25 and abs(measurement) < 0.5:
				samples.append(line)
				samples.append(line)
				samples.append(line)

			# HIgher steering angle have very little data. Add them more times
			elif abs(measurement) >= 0.5:
				samples.append(line)
				samples.append(line)
				samples.append(line)
				samples.append(line)

			# Remaining can be added single time as they have average numbers
			else:
				samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=BATCH_SIZE, training=True)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, training=True)

# save checkpoints after every epoch so that we can compare performance
checkpoint = ModelCheckpoint('model{epoch:02d}-{val_loss:.2f}.h5')

# training on multi GPU machine. 
# Use single GPU as other is used for other purposes
with K.tf.device('/gpu:0'):
	K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
	model = commaai_model()
	# model = nvidia_model()
	print(model.summary())

	# use Adam optimizer
	# model.compile(loss = 'mse', optimizer=Adam(lr=1e-5))
	model.compile(loss = 'mse', optimizer=Adam())
	model.fit_generator(train_generator, steps_per_epoch = \
		len(train_samples * 10 )/BATCH_SIZE, validation_data=validation_generator, \
		validation_steps=len(validation_samples * 10)/BATCH_SIZE, epochs=50, \
		callbacks=[checkpoint])

	model.save('model.h5')
