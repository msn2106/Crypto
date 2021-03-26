#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# In[6]:


data = pd.read_csv('BTC-USD.csv', date_parser=True)
data.tail()

# In[7]:


data_training = data[data['Date'] < '2020-01-01'].copy()
data_training

# In[8]:


data_test = data[data['Date'] > '2020-01-01'].copy()
data_test

# In[9]:


training_data = data_training.drop(['Date', 'Adj Close'], axis=1)
training_data.head()

# In[10]:


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
training_data

# In[11]:


X_train = []
Y_train = []

# In[12]:


training_data.shape[0]

# In[13]:


for i in range(60, training_data.shape[0]):
    X_train.append(training_data[i - 60:i])
    Y_train.append(training_data[i, 0])

# In[14]:


X_train, Y_train = np.array(X_train), np.array(Y_train)

# In[15]:


X_train.shape

# # Building LSTM

# In[16]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# In[78]:


regressor = Sequential()
regressor.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 5)))
regressor.add(Dropout(0.2))

# In[79]:


regressor.add(LSTM(units=60, activation='relu', return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=80, activation='relu', return_sequences=True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units=120, activation='relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units=1))

# In[80]:


regressor.summary()

# In[82]:


regressor.compile(optimizer='adam', loss='mean_squared_error')

# In[97]:


regressor.fit(X_train, Y_train, epochs=20, batch_size=50)

#  # Test Dataset

# In[100]:


past_60_days = data_training.tail(60)
df = past_60_days.append(data_test, ignore_index=True)
df = df.drop(['Date', 'Adj Close'], axis=1)
df.head()

# In[101]:


inputs = scaler.transform(df)
inputs

# In[102]:


X_test = []
Y_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i])
    Y_test.append(inputs[i, 0])

# In[103]:


X_test, Y_test = np.array(X_test), np.array(Y_test)
X_test.shape, Y_test.shape

# In[104]:


Y_pred = regressor.predict(X_test)
Y_pred, Y_test

# In[105]:


scaler.scale_

# In[106]:


scale = 1 / 5.18164146e-05
scale

# In[107]:


Y_test = Y_test * scale
Y_pred = Y_pred * scale

# In[108]:


Y_pred

# In[95]:


Y_test

# In[109]:


plt.figure(figsize=(14, 5))
plt.plot(Y_test, color='red', label='Real Bitcoin Price')
plt.plot(Y_pred, color='green', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction using RNN-LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# In[ ]:


# In[ ]:
