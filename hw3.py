import numpy as np
import pickle
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D

#config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))


class Model(object):
	def __init__(self):
		class_num = 10


		self.train_x = pickle.load(open("x_data", "rb"))
		self.train_y = pickle.load(open("y_data", "rb"))

		'''train_data = pickle.load(open("data/all_label.p","rb"))
		train_data = np.array(train_data)

		self.train_y = np.zeros((5000,10))
		self.train_x = np.zeros((3,32,32))
		self.train_x = np.reshape(self.train_x, [1,3,32,32])
		for i in range(10):
			for j in range(500):
				self.train_y[i*500+j][i] = 1

				cl_r = np.reshape(train_data[i][j][0:1024],[1,32,32])
				cl_g = np.reshape(train_data[i][j][1024:2048],[1,32,32])
				cl_b = np.reshape(train_data[i][j][2048:3072],[1,32,32])
				cl = np.concatenate((cl_r, cl_g, cl_b), axis=0)
				cl = np.reshape(cl, [1,3,32,32])
				self.train_x = np.concatenate((self.train_x, cl),axis=0)

		self.train_x = np.delete(self.train_x, 0, 0)

		pickle.dump(self.train_x, open("x_data", "wb"))
		pickle.dump(self.train_y, open("y_data", "wb"))'''


	def training(self):
		self.cnn_model = Sequential()
		
		'''self.cnn_model.add(Activation('relu'))
		self.cnn_model.add(Convolution2D(32, 3, 3, dim_ordering="th"))
		self.cnn_model.add(Activation('relu'))
		self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering="th"))

		self.cnn_model.add(Convolution2D(64, 3, 3, dim_ordering="th"))
		self.cnn_model.add(Activation('relu'))
		self.cnn_model.add(Convolution2D(64, 3, 3, dim_ordering="th"))
		self.cnn_model.add(Activation('relu'))
		self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering="th"))'''
		self.cnn_model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), dim_ordering = "th"))
		#self.cnn_model.add(Convolution2D(25, 3, 3, dim_ordering = "th"))
		self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering = "th")) 
		self.cnn_model.add(Convolution2D(64, 3, 3, dim_ordering = "th"))
		self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering = "th"))
		#self.cnn_model.add(Convolution2D(25, 3, 3, dim_ordering = "th"))
		#self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering = "th"))
		self.cnn_model.add(Flatten())
		self.cnn_model.add(Dense(output_dim = 1000))
		self.cnn_model.add(Activation('relu'))
		'''self.cnn_model.add(Dense(output_dim = 300))
		self.cnn_model.add(Activation('sigmoid'))
		self.cnn_model.add(Dense(output_dim = 300))
		self.cnn_model.add(Activation('sigmoid'))
		self.cnn_model.add(Dense(output_dim = 300))
		self.cnn_model.add(Activation('sigmoid'))'''
		self.cnn_model.summary()
		self.cnn_model.add(Dense(output_dim = 10))
		self.cnn_model.add(Activation('softmax'))



		self.cnn_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
		self.cnn_model.fit(self.train_x, self.train_y, batch_size=1000, nb_epoch=20)

			
def main():
	model = Model()
	model.training()

if __name__ == '__main__':
	main()



