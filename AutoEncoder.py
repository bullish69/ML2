import keras  
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.models import Model
from keras.layers import Input, Dense 
from sklearn import datasets 
from sklearn.preprocessing import MinMaxScaler 


iris = datasets.load_iris() 
X = iris.data 
Y = iris.target 
target_names = iris.target_names 


scaler = MinMaxScaler() 
scaler.fit(X) 
X_scaled = scaler.transform(X) 


def plot3clusters(X, title, vtitle): 
  plt.figure() 
  colors = ['red', 'green', 'blue'] 
  lw = 2 
  for color, i, target_name in zip(colors, [0, 1, 2], target_names): 
    plt.scatter(X[Y==i, 0], X[Y==i, 1], color=color, alpha=1., lw=lw, label=target_name) 
    plt.legend(loc='best', shadow=False, scatterpoints=1) 
    plt.title(title) 
    plt.xlabel(vtitle + "1") 
    plt.ylabel(vtitle + "2") 
    plt.show() 


input_dim = X_scaled.shape[1] 
encoding_dim = 2 
input_img = Input(shape=(input_dim, )) 
encoded = Dense(encoding_dim, activation='sigmoid')(input_img) 
decoded = Dense(input_dim, activation='sigmoid')(encoded) 
autoencoder = Model(input_img, decoded) 
autoencoder.compile(optimizer='adam', loss='mse') 
print(autoencoder.summary())


history = autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=16, shuffle=True, validation_split=0.1, verbose=1)



plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model Train vs. Validation Loss') 
plt.xlabel("Epoch") 
plt.ylabel("Loss") 
plt.legend(['loss', 'val_loss'], loc='upper right') 
plt.show()



# Use our encoded layer to encode the training input 
encoder = Model(input_img, encoded) 
encoded_input = Input(shape=(encoding_dim,)) 
decoder_layer = autoencoder.layers[-1] 
decoder = Model(encoded_input, decoder_layer(encoded_input)) 
encoded_data = encoder.predict(X_scaled) 
plot3clusters(encoded_data[:,:2], 'Linear AE', 'AE')