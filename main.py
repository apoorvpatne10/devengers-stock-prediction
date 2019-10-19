# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from keras.models import load_model

model = load_model('weights/my_model.h5')


data = pd.read_csv('data.csv')

data.dropna(inplace=True)

data.isnull().sum()

cl = data['High']

scl = MinMaxScaler()
cl = cl.values.reshape(cl.shape[0], 1)
cl = scl.fit_transform(cl)

#Create a function to process the data into 7 day look back slices
def processData(data, lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)

X,y = processData(cl, 7)

print(X[0])
print(y[0])


X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
print(X_train.shape[0])
print(X_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])

#Build the model
# model = Sequential()
# model.add(LSTM(256,input_shape=(7,1)))
# # model.add(LSTM(128))
# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mse')
#
# #Reshape data for (Sample,Timestep,Features)
# X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
# X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#
# #Fit model with history to check for overfitting
# history= model.fit(X_train,y_train,epochs=150,validation_data=(X_test,y_test),shuffle=False)

#model.save('my_model.h5')

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.show()

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))


Xt = model.predict(X_test)
Xt = scl.inverse_transform(Xt)

new_data = X_test[-1]
pred = Xt[-1]

threshold = 50200

#predx = scl.inverse_transform(pred)

import datetime
date = datetime.datetime(2019, 10, 19).date()

response = ''

for _ in range(10):
    new_data = np.append(new_data[1:], pred)
    new_datax = new_data.reshape(-1, 7, 1)
    pred = model.predict(new_datax)
    predx =  scl.inverse_transform(pred)[0][0]
    if predx > threshold:
        response += f"Stock price will be higher than threshold on {date.day}/{date.month}/{date.year}\n"
    else:
        response += f"Stock price is lower than threshold on {date.day}/{date.month}/{date.year}\n"
    print(f"Predicted price on {date.day}/{date.month}/{date.year} : {predx}")
    date += datetime.timedelta(days=1)


import send_sms
from send_sms import check_threshold

stuff = check_threshold(response)

plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
plt.plot(scl.inverse_transform(Xt))
plt.show()

#
# act = []
# pred = []
# for i in range(47):
#     Xt = model.predict(X_test[i].reshape(1,7,1))
#     print(f"predicted:{scl.inverse_transform(Xt)}, actual:{scl.inverse_transform(y_test[i].reshape(-1,1))}")
#     pred.append(scl.inverse_transform(Xt))
#     act.append(scl.inverse_transform(y_test[i].reshape(-1,1)))
#
# result_df = pd.DataFrame({'pred':list(np.reshape(pred, (-1))),'act':list(np.reshape(act, (-1)))})
#
# Xt = model.predict(X_test)
# plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
# plt.plot(scl.inverse_transform(Xt))
#
#
#
