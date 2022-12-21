from keras import backend as k
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

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
X_train /= 255
X_test /= 255

X_train.shape

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

print(np.unique(Y_train, return_counts=True))

num_classes = 10
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


def build_model(o):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='sigmoid'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    if o == 'sgd':
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='sgd', metrics=['accuracy'])
    elif o == 'adadelta':
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adadelta', metrics=['accuracy'])
    elif o == 'adagrad':
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adagrad', metrics=['accuracy'])
    elif o == 'adam':
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam', metrics=['accuracy'])
    elif o == 'rmsprop':
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='rmsprop', metrics=['accuracy'])

    model.summary()

    return model


"""# **SGD**"""

model_sgd = build_model('sgd')
model_log_sgd = model_sgd.fit(X_train, Y_train, batch_size=64,
                              epochs=10, verbose=1, validation_data=(X_test, Y_test))

scores_sgd = model_sgd.evaluate(X_test, Y_test, verbose=0)
print('SGD TEST LOSS:', scores_sgd[0])
print('SGD TEST ACCURACY:', scores_sgd[1])

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(model_log_sgd.history['accuracy'])
plt.plot(model_log_sgd.history['val_accuracy'])
plt.title("SGD Accuracy")
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(['train', 'test'], loc='best')

plt.subplot(2, 1, 2)
plt.plot(model_log_sgd.history['loss'])
plt.plot(model_log_sgd.history['val_loss'])
plt.title("SGD Loss")
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend(['train', 'test'], loc='best')

plt.tight_layout()

"""# **ADADELTA**"""

model_adadelta = build_model('adadelta')
model_log_adadelta = model_adadelta.fit(
    X_train, Y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, Y_test))

scores_adadelta = model_adadelta.evaluate(X_test, Y_test, verbose=0)
print('ADA DELTA TEST LOSS:', scores_adadelta[0])
print('ADA DELTA TEST ACCURACY:', scores_adadelta[1])

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(model_log_adadelta.history['accuracy'])
plt.plot(model_log_adadelta.history['val_accuracy'])
plt.title("ADA DELTA Accuracy")
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(['train', 'test'], loc='best')

plt.subplot(2, 1, 2)
plt.plot(model_log_adadelta.history['loss'])
plt.plot(model_log_adadelta.history['val_loss'])
plt.title("ADA DELTA Loss")
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend(['train', 'test'], loc='best')

plt.tight_layout()

"""# **ADAGRAD**"""

model_adagrad = build_model('adagrad')
model_log_adagrad = model_adagrad.fit(
    X_train, Y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, Y_test))

scores_adagrad = model_adagrad.evaluate(X_test, Y_test, verbose=0)
print('ADAGRAD TEST LOSS:', scores_adagrad[0])
print('ADAGRAD TEST ACCURACY:', scores_adagrad[1])

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(model_log_adagrad.history['accuracy'])
plt.plot(model_log_adagrad.history['val_accuracy'])
plt.title("ADAGRAD Accuracy")
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(['train', 'test'], loc='best')

plt.subplot(2, 1, 2)
plt.plot(model_log_adagrad.history['loss'])
plt.plot(model_log_adagrad.history['val_loss'])
plt.title("ADAGRAD Loss")
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend(['train', 'test'], loc='best')

plt.tight_layout()

"""# **ADAM**"""

model_adam = build_model('adam')
model_log_adam = model_adam.fit(
    X_train, Y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, Y_test))

scores_adam = model_adam.evaluate(X_test, Y_test, verbose=0)
print('ADAM TEST LOSS:', scores_adam[0])
print('ADAM TEST ACCURACY:', scores_adam[1])

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(model_log_adam.history['accuracy'])
plt.plot(model_log_adam.history['val_accuracy'])
plt.title("ADAM Accuracy")
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(['train', 'test'], loc='best')

plt.subplot(2, 1, 2)
plt.plot(model_log_adam.history['loss'])
plt.plot(model_log_adam.history['val_loss'])
plt.title("ADAM Loss")
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend(['train', 'test'], loc='best')

plt.tight_layout()

"""# **RMS PROP**"""

model_rms = build_model('rmsprop')
model_log_rms = model_rms.fit(X_train, Y_train, batch_size=64,
                              epochs=10, verbose=1, validation_data=(X_test, Y_test))

scores_rms = model_rms.evaluate(X_test, Y_test, verbose=0)
print('RMS TEST LOSS:', scores_rms[0])
print('RMS TEST ACCURACY:', scores_rms[1])

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(model_log_rms.history['accuracy'])
plt.plot(model_log_rms.history['val_accuracy'])
plt.title("RMS Accuracy")
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY")
plt.legend(['train', 'test'], loc='best')

plt.subplot(2, 1, 2)
plt.plot(model_log_rms.history['loss'])
plt.plot(model_log_rms.history['val_loss'])
plt.title("RMS Loss")
plt.xlabel("EPOCH")
plt.ylabel("LOSS")
plt.legend(['train', 'test'], loc='best')

plt.tight_layout()

"""# **COMPARISON**"""

plt.figure(figsize=(10, 10))
plt.plot(model_log_sgd.history['accuracy'])
plt.plot(model_log_adadelta.history['accuracy'])
plt.plot(model_log_adagrad.history['accuracy'])
plt.plot(model_log_adam.history['accuracy'])
plt.plot(model_log_rms.history['accuracy'])
plt.title("TRAIN ACCURACY")
plt.xlabel("EPOCHS")
plt.ylabel("TRAINING ACCURACY")
plt.legend(['SGD', 'ADA DELTA', 'ADA GRAD', 'ADAM', 'RMSPROP'], loc='best')

plt.figure(figsize=(10, 10))
plt.plot(model_log_sgd.history['val_accuracy'])
plt.plot(model_log_adadelta.history['val_accuracy'])
plt.plot(model_log_adagrad.history['val_accuracy'])
plt.plot(model_log_adam.history['val_accuracy'])
plt.plot(model_log_rms.history['val_accuracy'])
plt.title("TEST ACCURACY")
plt.xlabel("EPOCHS")
plt.ylabel("TESTING ACCURACY")
plt.legend(['SGD', 'ADA DELTA', 'ADA GRAD', 'ADAM', 'RMSPROP'], loc='best')
