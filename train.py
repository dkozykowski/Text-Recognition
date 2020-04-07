import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist

# reading data from keras mnist:
# X_train - picture of handwritten digit used for training
# Y_train - name of that digit
# X_test - picture of handwritten digit used for testing effects of learning
# Y_test - name of that digit
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# single mnist data format
height = 28
width = 28
input_shape = (height, width, 1)

# convert given image to straight vector
X_train = X_train.reshape(X_train.shape[0], height, width, 1)
X_test = X_test.reshape(X_test.shape[0], height, width, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-hot encoding
typesNumber = 10
y_train = keras.utils.to_categorical(y_train, typesNumber)
y_test = keras.utils.to_categorical(y_test, typesNumber)

# adding hidden layers to the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(typesNumber, activation='softmax'))

# compiling the sequential model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# training the model
model.fit(X_train,
          y_train,
          batch_size=128,
          epochs=1,
          verbose=1,
          validation_data=(X_test, y_test))

# showing results
result = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy:', int(result[1] * 100), "%")
# model.save("model_v1.0")
