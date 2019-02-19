#!/usr/bin/env python
# coding: utf-8

# In[251]:


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
# Le but de ce notebook ( suite du notebook feature_generation_1 ) est de préparer les données pour qu'elle puie être ingérer par un algo d'apprentissage.

# In[252]:


client_ori = pd.read_pickle("./client_df.pkl")


# In[253]:


client_ori.head()


# In[254]:


client = client_ori.copy()
client.head()


# # 1. Remove useless column

# In[255]:


client = client.drop(['dep', 'id_cmde_sous_crypt','id_cmde_sous_crypt','date_fin_essai'], axis=1)


# # 2. Convert date to timestamp

# In[256]:


client.head()


# In[257]:


def to_timestamp(date):

    date = datetime.datetime.strptime(date, "%d/%m/%Y").date()
    date = date.strftime("%s")
    return int(date)


# In[258]:


client['fld_account_fcom_create_date'] = client.apply(lambda row : to_timestamp(row['fld_account_fcom_create_date']), axis = 1) 
client['date_souscription_essai'] = client.apply(lambda row : to_timestamp(row['date_souscription_essai']), axis = 1) 


# In[259]:


client.head()


# # 3. Missing value

# In[260]:


client.isna().sum()[client.isna().sum() != 0]


# In[261]:


client.year_birthdate.mode()


# In[262]:


client.year_birthdate.fillna(client.year_birthdate.mode()[0],inplace=True)


# In[263]:


client.isna().sum()[client.isna().sum() != 0]


# # 4. Hot encoding

# In[264]:


target = client[['id_fcom_sha256','top_converti']]
predictors_quanti = client.drop(['top_converti','fld_acc_gender_id_fk','fld_email_optin_fk'], axis=1)
predictors_quali = client[['id_fcom_sha256','fld_acc_gender_id_fk','fld_email_optin_fk']]


# In[265]:


## One hot encode the columns of the database specified in list_id.
#
#  return a copy of the database, on hot encoded. 
def one_hot_encode_database(database, list_id):
    
    col_names_ori = list(database.columns.values)
    database = database.values
    
    col_names = []
    
    encoded_database = np.empty(shape=(database.shape[0],0), dtype=float)
    
    for id in range(database.shape[1]):
        if id in list_id:
            
            original_column = database[:, id]
            encoded_column = to_categorical(original_column)
            encoded_database = np.column_stack((encoded_database,encoded_column))
            
            for i in range(encoded_column.shape[1]):
                col_names.append(col_names_ori[id] + '_' + str(i))
            
        else:
            original_column = database[:, id]
            encoded_database = np.column_stack((encoded_database,original_column))
            col_names.append(col_names_ori[id])
        
        encoded_database = pd.DataFrame(data=encoded_database,
            columns=col_names)

        
    return encoded_database

## Normalize between 0 and 1 each column of the database specified in list_id.
#
#  return a copy of the database, normalized. 
def normalize_database(database, list_id):
    
    
    col_names_ori = list(database.columns.values)
    database = database.values
    
    encoded_database = database.copy()
    for  id in list_id :
        x = encoded_database[:, id]
        encoded_database[:, id] = (x-min(x))/(max(x)-min(x))
        
    encoded_database = pd.DataFrame(data=encoded_database,
    columns=col_names_ori)
        
    return encoded_database


# In[266]:


predictors_quali = one_hot_encode_database(predictors_quali,range(predictors_quali.shape[1])[1:])
target = one_hot_encode_database(target,range(target.shape[1])[1:])


# # 5. Normalise database

# In[267]:


predictors_quanti = normalize_database(predictors_quanti,range(predictors_quanti.shape[1])[1:])


# In[268]:


predictors = pd.merge(predictors_quanti, predictors_quali, on='id_fcom_sha256', how='inner')


# In[269]:


predictors.to_pickle("./predictors.pkl")
target.to_pickle("./target.pkl")


# In[ ]:




