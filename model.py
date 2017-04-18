import csv
import os, os.path
import cv2
import numpy as np
import tensorflow as tf

# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import ELU, Lambda, Cropping2D
from keras import regularizers
from keras.callbacks import ModelCheckpoint, Callback
import keras.backend.tensorflow_backend as K

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS
BATCH_SIZE = 32
#steering_offset = 0.25
translation_offset = 0.01
steering_offset = 0.1595

def get_all_images(sample, lines):
	#lines = []

	measurement = float(line[3].strip())
	if abs(measurement) <= 0.01:
		prob = np.random.random()
		if prob < 0.2:
			return lines

	img_path = os.path.join("data", line[0].strip())
	if os.path.isfile(img_path):
		lines.append([line[0], measurement])
		#measurements.append(measurement)
	else:
		print("Missing file {}".format(img_path))

	left_cam_img_path = os.path.join("data", line[1].strip())
	if os.path.isfile(left_cam_img_path):
		measurement_left = measurement + steering_offset
		lines.append([line[1], measurement_left])
		#measurements.append(measurement_left)
	else:
		print("Missing file {}".format(left_cam_img_path))

	right_cam_img_path = os.path.join("data", line[2].strip())
	if os.path.isfile(right_cam_img_path):
		measurement_right = measurement - steering_offset
		lines.append([line[2], measurement_right])
		#measurements.append(measurement_right)
	else:
		print("Missing file {}".format(right_cam_img_path))

	return lines

def image_translation(image, angle):
	# Compute X translation
	x_translation = np.random.randint(-20, 20)
	if(x_translation < 0):
		angle = angle + x_translation * translation_offset
	else:
		angle = angle - x_translation * translation_offset
	# Randomly compute a Y translation
	y_translation = 0
	# Form the translation matrix
	translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
	# Translate the image
	image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
	return image, angle


def random_modify(image, angle):
	oper = np.random.randint(0,5)
	if(oper == 0):
		image = cv2.blur(image, (5,5))
	elif(oper == 1):
		kernel = np.ones((5,5),np.uint8)
		image = cv2.erode(image,kernel,iterations = 1)
	elif(oper == 2):
		rows, cols, chnls = image.shape
		mean = 0
		gauss = np.random.normal(mean, 1, (rows, cols, chnls))
		gauss = gauss.reshape(rows, cols, chnls)
		image = image + gauss

	#image, angle = image_translation(image, angle)
	return image, angle


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
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					
					flip_prob = np.random.uniform()
					if flip_prob < 0.5 :
						image = cv2.flip(image, 1)
						measurement = -measurement

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

def commaai_model():
	input_shape=(160, 320, 3)
	model = Sequential()
	model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=input_shape))
	#model.add(Lambda(lambda image: K.tf.image.resize_images(image, (64, 64))))
	model.add(Lambda(lambda image: K.tf.image.rgb_to_hsv(image)))
	model.add(Lambda(lambda x: (x / 255.0) - 0.5))
	# model.add(Conv2D(16, kernel_size = 3, strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(0.01)))
	# model.add(ELU())
	# model.add(Conv2D(32, kernel_size=3, strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(0.01)))
	# model.add(ELU())
	# model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(0.01)))
	# model.add(Flatten())
	# model.add(Dropout(.2))
	# model.add(ELU())
	# model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
	# model.add(Dropout(.5))
	# model.add(ELU())
	# # model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01)))
	# # model.add(Dropout(.5))
	# # model.add(ELU())
	# model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01)))


	model.add(Conv2D(16, kernel_size=8, strides=(4, 4), padding="same", kernel_regularizer=regularizers.l2(0.01)))
	model.add(ELU())
	model.add(Conv2D(32, kernel_size=5, strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(0.01)))
	model.add(ELU())
	model.add(Conv2D(64, kernel_size=5, strides=(2, 2), padding="same", kernel_regularizer=regularizers.l2(0.01)))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
	return model

samples = []
with open('../data/udacity_driving_log_enhanced.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		#samples = get_all_images(line, samples)
		samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=BATCH_SIZE, training=False)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE, training=False)
checkpoint = ModelCheckpoint('model{epoch:02d}-{val_loss:.2f}.h5')

with K.tf.device('/gpu:0'):
	K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
	model = commaai_model()
	print(model.summary())
	model.compile(loss = 'mse', optimizer = 'Adam')
	model.fit_generator(train_generator, steps_per_epoch = \
		1000, validation_data=validation_generator, \
		validation_steps=len(validation_samples)/BATCH_SIZE, epochs=5, \
		callbacks=[checkpoint])
	# model.fit_generator(train_generator, steps_per_epoch = \
	# 	len(train_samples)/BATCH_SIZE, validation_data=validation_generator, \
	# 	validation_steps=len(validation_samples)/BATCH_SIZE, epochs=5, \
	# 	callbacks=[checkpoint])

	model.save('model.h5')
