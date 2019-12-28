import csv #to access *.csv file
import cv2
import numpy as np
import tensorflow as tf
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras import regularizers
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers import Input, Dense, Flatten, Lambda, Activation, Dropout, ELU

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('epochs', 5, "Number of epochs used for training")
flags.DEFINE_string('batch_size', 32, "Batch size used for training")
flags.DEFINE_string('test_size', 0.2, "Proportion of the dataset to include in the test split")

#load data from CSV file
def load_data():
	samples = []
	with open("./data/driving_log.csv") as csvfile:
		content = csv.reader(csvfile)
		 # Remove header from csvfile
		next(content, None)
		for line in content:
			samples.append(line)

	train_samples, validation_samples = train_test_split(samples, test_size=float(FLAGS.test_size))
	return (train_samples, validation_samples)
#cv2.imread will get images in BGR format, while drive.py uses RGB. So, this definition will convert bgr to rgb format
def bgr_to_rgb(img):
	b,g,r = cv2.split(img) # get b,g,r
	return cv2.merge([r,g,b]) # switch it to rgb

def generator(samples, batch_size):
	num_samples = len(samples)
	while(1): # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = list()
			steering_angles = list()
			for batch_sample in batch_samples:
				# Skip it if very low speed - not representative of driving behavior
				if (float(batch_sample[6]) >= 0.1):
					steering_center = float(batch_sample[3])

					# Create adjusted steering measurements for the side camera images
					correction = 0.25
					steering_left = steering_center + correction
					steering_right = steering_center - correction

					# Change image paths as the learning has been done elsewhere
					path = "./data/IMG/"
					img_center = bgr_to_rgb(cv2.imread(path + batch_sample[0].split('/')[-1]))
					img_left = bgr_to_rgb(cv2.imread(path + batch_sample[1].split('/')[-1]))
					img_right = bgr_to_rgb(cv2.imread(path + batch_sample[2].split('/')[-1]))

					# Load images and steering angles
					images.extend([img_center, img_left, img_right])
					steering_angles.extend([steering_center, steering_left, steering_right])

					# Augment data by flipping image around y-axis
					aug_img_center = cv2.flip(img_center, 1)
					aug_img_left = cv2.flip(img_left, 1)
					aug_img_right = cv2.flip(img_right, 1)
					images.extend([aug_img_center, aug_img_left, aug_img_right])
					steering_angles.extend([-1.0*steering_center, -1.0*steering_left, -1.0*steering_right])

			# Return numpy arrays
			X_train = np.array(images)
			y_train = np.array(steering_angles)
			yield shuffle(X_train, y_train)

def grayscale(input):
	from keras.backend import tf as ktf
	return ktf.image.rgb_to_grayscale(input)

def resize_image(input, h, w):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(input, [h,w], ktf.image.ResizeMethod.BICUBIC)

def normalize_image(input):
	return (input / 255.0) - 0.5

def build_lenet_model(input_shape):
	model = Sequential()

	model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=input_shape))# --- Crop image to save only the region of interest
	model.add(Lambda(grayscale))# --- Convert image into grayscale
	model.add(Lambda(resize_image, arguments={'h': 32, 'w': 32}))# --- Resize it to have a 32x32 shape
	model.add(Lambda(normalize_image))# --- Normalize and mean center the data

	# --- Layer 1 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

	# --- Layer 2 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

	# --- Flatten the weights
	model.add(Flatten())

	# --- Layer 3 : Fully-connected + ReLu activation
	model.add(Dense(120, activity_regularizer=regularizers.l2(0.01)))
	model.add(Activation('relu'))

	# --- Layer 4 : Fully-connected + ReLu activation
	model.add(Dense(84, activity_regularizer=regularizers.l2(0.01)))
	model.add(Activation('relu'))

	# --- Layer 5 : Fully-connected
	model.add(Dense(1))

	return model

def build_nvidia_model(input_shape):
	model = Sequential()

	model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=input_shape))# --- Crop image to save only the region of interest
	model.add(Lambda(normalize_image))# --- Normalize and mean center the data

	# --- Layer 1 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2,2)))

	# --- Layer 2 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2,2)))

	# --- Layer 3 : Convolution + ReLu activation + maxpooling
	model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())
	model.add(MaxPooling2D(pool_size=(2,2)))

	# --- Layer 4 : Convolution + ReLu activation
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Layer 5 : Convolution + ReLu activation
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Flatten the weights
	model.add(Flatten())

	# --- Layer 6 : Fully-connected + ReLu activation
	model.add(Dense(100, kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Layer 7 : Fully-connected + ReLu activation
	model.add(Dense(50, kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Layer 8 : Fully-connected + ReLu activation
	model.add(Dense(10, kernel_regularizer=regularizers.l2(0.001)))
	model.add(ELU())

	# --- Layer 9 : Fully-connected
	model.add(Dense(1))

	return model

def main(_):
	# Import training and validation data and create generators
	train_samples, validation_samples = load_data()
	train_generator = generator(train_samples, batch_size=int(FLAGS.batch_size))
	validation_generator = generator(validation_samples, batch_size=int(FLAGS.batch_size))

	# Build the model
	model = build_nvidia_model((160,320,3))
	print(model.summary())
	# Compile and train the model using the generator function
	model.compile(loss='mse', optimizer=Adam(lr=1e-4))
	history_object = model.fit_generator(train_generator,\
		steps_per_epoch=len(train_samples)/int(FLAGS.batch_size),\
		epochs=int(FLAGS.epochs),\
		validation_data=validation_generator,\
		validation_steps=len(validation_samples)/int(FLAGS.batch_size),\
		verbose=1)

	print(history_object.history.keys())

	# Plot the training and validation loss after each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('Model mean squared error loss')
	plt.ylabel('Mean squared error loss')
	plt.xlabel('Epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()
	plt.savefig('training_validation_loss.png')

	# Save it
	model.save('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()