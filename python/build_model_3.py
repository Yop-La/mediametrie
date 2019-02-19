#!/usr/bin/env python
# coding: utf-8

# In[149]:


import pandas as pd
import numpy as np
import datetime
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from keras.utils import to_categorical
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, rmsprop, sgd
from keras.callbacks import EarlyStopping
import keras.backend as Ka
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold
from numpy import array
from keras import optimizers
from sklearn.model_selection import train_test_split


# # Contexte 
# 
# Le but de ce notebook est d'élaborer un modèle capable de prédire la variable top_converti ( suite du notebook prepare_data_2 ).
# 
# Pour cela, nous allons utiliser un réseau de neurones à l'aide de la libraire Keras
# 
# En terme de performance, le modèle entrainé ici a une précision de 98.2 % sur le jeu de test composé de 9900 observations.

# In[150]:


predictors_ori = pd.read_pickle("./predictors.pkl")
target_ori = pd.read_pickle("./target.pkl")


# In[151]:


predictors = predictors_ori.copy()
target = target_ori.copy()


# # 1. Deux fonctions utiles
# 
# La première permet de voir l'évolution de la loss et de l'accuracy sur le jeu de test et de validation pendant l'apprentissage. Elle nous permet de diagnostiquer un éventuel sur-apprentissage.
# 
# La deuxième permet de construire et de compiler un réseau de neurones

# In[152]:


def plotLog(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# In[153]:


## Construct a layer composed of dense layers, which dimensions are definded in the layer_list argument.
# 
#  return the constructed and compiled model.
def build_NN(layer_list, input_dim, output_dim, lr=0.001):
    model = Sequential();
    
    for idx, layer in enumerate(layer_list):
        if(idx == 0):
            model.add(Dense(layer,activation='relu',input_shape=(input_dim,)))
        else:
            model.add(Dense(layer,activation='relu'))

    model.add(Dense(output_dim,activation='softmax'))
    
    sgd = optimizers.SGD(lr=lr)
    #Compile the network
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# # 2. Jeu de test et d'entraînement

# In[154]:


X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(predictors, target, test_size=0.33, random_state=42)


# In[155]:


X_train = X_train_df.values
X_test = X_test_df.values
y_train = y_train_df.values
y_test = y_test_df.values


# # 3. Entraînement du modèle

# In[156]:


layer_list = [1000,100]
input_dim = X_train.shape[1]-1
output_dim = y_train.shape[1]-1
lr = 0.01
epochs = 10000


# In[157]:


model = build_NN(layer_list, input_dim, output_dim, lr=lr)
callback = EarlyStopping(patience=10)
history = model.fit(X_train[:,1:], y_train[:,1:], batch_size=16, epochs=epochs, validation_data=(X_test[:,1:], y_test[:,1:]), shuffle=True, callbacks=[callback])


# In[158]:


model.save('my_model.h5')


# In[159]:


plotLog(history)


# # 4. Performances du modèle

# In[160]:


test_df = pd.merge(X_test_df, y_test_df, on='id_fcom_sha256', how='inner')


# In[161]:


predictions = model.predict(test_df.drop(['id_fcom_sha256','top_converti_0','top_converti_1'],axis=1))


# In[164]:


# tableau qui va contenir les valeurs réelles et les prédictions
gt_and_pred = np.empty(shape=(2,0)) 
gt_and_pred = np.column_stack((gt_and_pred, np.array([np.argmax(test_df[['top_converti_0','top_converti_1']].values, axis=1), np.argmax(predictions, axis=1)])))


# In[165]:


kappa = cohen_kappa_score(gt_and_pred[0], gt_and_pred[1], labels=[0,1])


# In[166]:


kappa


# In[167]:


conf_mat = confusion_matrix(gt_and_pred[0], gt_and_pred[1], labels=[0,1])


# In[168]:


conf_mat


# In[169]:


precision = (conf_mat[0,0]+conf_mat[1,1])/(conf_mat.sum())


# In[170]:


precision

