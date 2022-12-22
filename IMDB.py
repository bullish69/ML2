import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

top_words = 5000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)

max_review_length = 50

X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)


X_train.shape


print(X_train[1])


embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words+1, embedding_vector_length,
          input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())


hist = model.fit(X_train, Y_train, batch_size=64, epochs=10,
                 verbose=1, validation_data=(X_test, Y_test))


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
plt.subplot(2, 1, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend(['loss', 'val_loss'], loc='lower right')
