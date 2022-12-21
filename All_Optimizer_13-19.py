import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

X = np.array([0.5, 2.5]) 
Y = np.array([0.2, 0.9])

def sigmoid(x, w, b): 
  y_in = np.dot(w, x) + b 
  y_hat = 1/(1 + np.exp(-y_in)) 
  return y_hat

def error(y, y_hat): 
  err = np.array((y-y_hat)**2).mean() 
  return err

def delta_w(x, y, y_hat, c): 
  dw = c*(y_hat-y)*y_hat*(1-y_hat)*x
  return dw

def delta_b(y, y_hat, c): 
  db = c*(y_hat-y)*y_hat*(1-y_hat)
  return db

"""# **BATCH**"""

def batch_gd(x, Y, epochs): 
  w = -2 
  b = -2 
  c = 1 
  err_list = [] 
  W = [] 
  B = [] 
  for i in range(epochs): 
    temp_w = 0 
    temp_b = 0
    for x, y in zip(X, Y):
      y_hat = sigmoid(x, w, b) 
      temp_w += delta_w(x, y, y_hat, c) 
      temp_b += delta_b(y, y_hat, c) 
    temp_w = temp_w/len(Y) 
    temp_b = temp_b/len(Y) 
    w -= temp_w 
    b -= temp_b 
    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight ==> {w} and Bias ==> {b}") 
  return W, B, err_list

wt_bgd, b_bgd, err_bgd = batch_gd(X, Y, 100)

epoch = [i for i in range(1, 101)]

plt.plot(epoch, err_bgd) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_bgd, err_bgd) 
plt.xlabel("Weight") 
plt.ylabel("Error") 
plt.show()

"""# **MINI-BATCH**"""

def mini_batch_gd(X, Y, epochs): 
  batch_size = 1 
  w = -2 
  b = -2 
  c = 1 
  err_list = [] 
  W = [] 
  B = [] 
  for i in range(epochs): 
    temp_dw = 0 
    temp_db = 0 
    counter = 0
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, w, b) 
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
      counter += 1 
      if(counter % batch_size == 0):  
        w -= temp_dw 
        b -= temp_db 
        temp_dw = 0 
        temp_db = 0  
    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight ==> {w} and Bias ==> {b}") 
  
  return W, B, err_list

wt_mb, bias_mb, err_mb = mini_batch_gd(X, Y, 100)

plt.plot(epoch, err_mb) 
plt.xlabel("Epochs") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_mb, err_mb) 
plt.xlabel("Weights") 
plt.ylabel("Error") 
plt.show()

"""# **MOMENTUM**"""

def momentum_gd(X, Y, epochs): 
  w = -2
  b = -2
  eta = 1
  c = 1
  gamma = 0.9
  v_w = 0 
  v_b = 0 
  err_list = [] 
  W = [] 
  B = [] 
  for i in range(epochs): 
    temp_dw = 0
    temp_db = 0 
    for x, y in zip(X, Y):  
      y_hat = sigmoid(x, w, b)
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
    v_w = gamma*v_w + eta*temp_dw 
    v_b = gamma*v_b + eta*temp_db 
    w -= v_w 
    b -= v_b 
    W.append(w) 
    B.append(b) 
    y_hat = sigmoid(X, w, b)
    err_list.append(error(Y, y_hat)) 
    print(f"After epoch {i+1}: Weight is {w} and Bias is {b}") 
  return W, B, err_list

wt_mom, bias_mom, err_mom = momentum_gd(X, Y, 100)

epoch = [i for i in range(1, 101)]

plt.plot(epoch, err_mom) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_mom, err_mom) 
plt.xlabel("Weight") 
plt.ylabel("Error") 
plt.show()

"""# **ADAGRAD**"""

def adagrad_gd(X, Y, epochs): 
  w = -2 
  b = -2 
  c = 1 
  eta = 1 
  eps = 2 
  err_list = [] 
  W = [] 
  B = [] 
  v_w = 0 
  v_b = 0 
  for i in range(epochs): 
    temp_dw = 0 
    temp_db = 0 
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, w, b)
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 

    v_w = v_w + temp_dw**2 
    v_b = v_b + temp_db**2 

    w = w - (eta*temp_dw)/(np.sqrt(v_w + eps)) 
    b = b - (eta*temp_db)/(np.sqrt(v_w + eps)) 

    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b))) 
    print(f"After epoch {i+1}: Weight is {w} and Bias is {w}") 
  return W, B, err_list

wt_adagrad, bias_adagrad, err_adagrad = adagrad_gd(X, Y, 100)

plt.plot(epoch, err_adagrad) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_adagrad, err_adagrad) 
plt.xlabel("Weight") 
plt.ylabel("Error") 
plt.show()

"""# **ADADELTA / RMSPROP**"""

def adadelta(X, Y, epochs): 
  w = -2 
  b = -2 
  c = 1 
  beta = 0.9
  eta = 1 
  eps = 2  
  vw = 0 
  vb = 0 
  W = [] 
  B = [] 
  err_list = [] 
  for i in range(epochs): 
    temp_dw = 0 
    temp_db = 0 
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, w, b) 
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
    
    vw = beta*vw + (1-beta)*temp_dw*temp_dw 
    vb = beta*vb + (1-beta)*temp_db*temp_db 

    w = w - (eta*temp_dw)/(np.sqrt(vw) + eps) 
    b = b - (eta*temp_db)/(np.sqrt(vb) + eps) 

    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b))) 
    print(f"After epoch {i+1}: Weight is {w} and Bias is {b}") 
  return W, B, err_list

wt_adadelta, bias_adadelta, err_adadelta = adadelta(X, Y, 100)

plt.plot(epoch, err_adadelta) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_adadelta, err_adadelta) 
plt.xlabel("Weights") 
plt.ylabel("Error") 
plt.show()

"""# **NAG**"""

def nag_gd(X, Y, epochs): 
  w = -2 
  b = -2 
  c = 1 
  eta = 1 
  beta = 0.9 
  W = [] 
  B = [] 
  err_list = [] 
  prev_vw = 0 
  prev_vb = 0 
  for i in range(epochs): 
    temp_dw = 0
    temp_db = 0 
    v_w = w - beta*prev_vw  
    v_b = b - beta*prev_vb 
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, v_w, v_b) 
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
    v_w = beta*prev_vw + eta*temp_dw 
    v_b = beta*prev_vb + eta*temp_db 

    w = w - v_w 
    b = b - v_b 
    prev_vw = v_w 
    prev_vb = v_b 
    W.append(w) 
    B.append(b) 
    err_list.append(error(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight is {w} and Bias is {b}") 
  return W, B, err_list

wt_nag, bias_nag, err_nag = nag_gd(X, Y, 100)

plt.plot(epoch, err_nag) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_nag, err_nag) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

"""# **ADAM**"""

def adam_gd(X, Y, epochs): 
  w, b, c = -2, -2, 1 
  beta1, beta2 = 0.45, 0.85
  eta = 1 
  mt_w, mt_b = 0, 0 
  vt_w, vt_b = 0, 0
  eps = 2 
  err_list = [] 
  W = [] 
  B = [] 
  for i in range(1, epochs+1): 
    temp_dw = 0 
    temp_db = 0 
    for x, y in zip(X, Y): 
      y_hat = sigmoid(x, w, b)
      temp_dw += delta_w(x, y, y_hat, c) 
      temp_db += delta_b(y, y_hat, c) 
    
    mt_w = beta1*mt_w + eta*temp_dw
    mt_b = beta1*mt_b + eta*temp_db 

    vt_w = beta2*vt_w + (1-beta2)*temp_dw*temp_dw
    vt_b = beta2*vt_b + (1-beta2)*temp_db*temp_db 

    mt_hat_w = mt_w/(1-beta1**i) 
    vt_hat_w = vt_w/(1-beta2**i) 
    w = w - (eta*mt_hat_w)/(np.sqrt(vt_hat_w) + eps)

    mt_hat_b = mt_b/(1-beta1**i) 
    vt_hat_b = vt_b/(1-beta2**i) 
    b = b - (eta*mt_hat_b)/(np.sqrt(vt_hat_b) + eps)

    W.append(w) 
    B.append(b) 

    err_list.append(error(Y, sigmoid(X, w, b)))
    print(f"After epoch {i+1}: Weight is {w} and Bias is {b}") 
  return W, B, err_list

wt_adam, bias_adam, err_adam = adam_gd(X, Y, 100)

plt.plot(epoch, err_adam) 
plt.xlabel("Epoch") 
plt.ylabel("Error") 
plt.show()

plt.plot(wt_adam, err_adam) 
plt.xlabel("Weight") 
plt.ylabel("Error") 
plt.show()

"""# **ANALYSIS OF ALL GRADIENTS**"""

plt.figure(figsize=(10, 10)) 
plt.plot(epoch, err_bgd, color='orange') 
plt.plot(epoch, err_mb, color='yellow')
plt.plot(epoch, err_mom, color='pink') 
plt.plot(epoch, err_adagrad, color='red')
plt.plot(epoch, err_adadelta, color='green') 
plt.plot(epoch, err_nag, color='blue')
plt.plot(epoch, err_adam, color='cyan') 
plt.legend(['Batch', 'Mini-Batch', 'Momentum', 'AdaGrad', 'AdaDelta', 'NAG', 'ADAM'], loc='best') 
plt.show()

