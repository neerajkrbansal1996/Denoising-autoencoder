from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model 
from keras import backend as K

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

# #this model maps an input to its encoded representation
# encoder = Model(input_img, encoded)

# #Create a placeholder for an encoded (32- dimensional) Input
# encoded_input = Input(shape=(28,28,1))
# # retrive the last layer of the autoencoded model
# deco = autoencoder.layers[-3](encoded_input)
# deco = autoencoder.layers[-2](deco)
# deco = autoencoder.layers[-1](deco)
# #Create the decoder model
# decoder = Model(encoded_input,decoded)

autoencoder.compile(optimizer='adadelta', loss = 'binary_crossentropy')


from keras.datasets import mnist
import numpy as np 
import matplotlib.pyplot as plt 


(x_train, _),(x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') /255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# n = 10
# plt.figure(figsize=(20, 2))
# for i in range(n):
#     ax = plt.subplot(1, n, i+1)
#     plt.imshow(x_test_noisy[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


autoencoder.fit(x_train_noisy, x_train, epochs=1, batch_size=128,shuffle=True,validation_data=(x_test_noisy, x_test))

# encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize = (20,4))
for i in range(n):
	#Display original
	ax = plt.subplot(2, n, i+1)
	plt.imshow(x_test_noisy[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	#Display Reconstruction
	ax = plt.subplot(2, n, i+1+n)
	plt.imshow(decoded_imgs[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()

