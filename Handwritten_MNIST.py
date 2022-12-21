from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D


import keras
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


fig = plt.figure(figsize=(10, 10))
for i in range(1, 16):
    rn = np.random.randint(60000)
    fig.add_subplot(3, 5, i)
    plt.imshow(X_train[rn], cmap='gray')
    plt.xlabel(Y_train[rn], color='g')
plt.show()


img_rows, img_cols = 28, 28


if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255


print('X_train shape:', X_train.shape)


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


print(np.unique(Y_train, return_counts=True))


num_classes = 10
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


num_epoch = 10
batch_size = 64

model = build_model()
model_log = model.fit(X_train, Y_train, batch_size=64,
                      epochs=num_epoch, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('ADAM Model Accruacy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('ADAM Model Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train', 'test'], loc='lower right')

plt.tight_layout()


def build_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='Adagrad', metrics=['accuracy'])

    model.summary()

    return model


num_epoch = 10
batch_size = 64

model = build_model()
model_log = model.fit(X_train, Y_train, batch_size=64,
                      epochs=num_epoch, verbose=1, validation_data=(X_test, Y_test))


score = model.evaluate(X_test, Y_test, verbose=0)
print('Adagrad Test Loss:', score[0])
print('Adagrad Test Accuracy:', score[1])


fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('AdaGrad Model Accruacy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('AdaGrad Model Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train', 'test'], loc='lower right')

plt.tight_layout()
