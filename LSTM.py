import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model


def create_dataset(dataset, time_step):
    X_data, Y_data = [], []
    for i in range(time_step,len(dataset)):
        #a = dataset[i:(i+time_step), 0]
        X_data.append(dataset[i-time_step:i , 0])
        Y_data.append(dataset[i , 0])
        #return X_data, Y_data
    return np.array(X_data), np.array(Y_data)

df = pd.read_csv('C:\Master\FB.csv', names=['Month','value'], header=0, index_col=0,parse_dates = [0])
#print(df)
scaler=MinMaxScaler(feature_range=(0,1))
df_scale = scaler.fit_transform(np.array(df))
#print(df_scale)
#print(df_scale.shape)

train_data = df_scale[:1285]
test_data = df_scale[1285:]

time_step = 40

X_train , y_train = create_dataset(train_data, time_step)
#create_dataset(test_data, time_step)
print("X" ,X_train.shape)
print("y" ,y_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#print(X_train)
#print(X_train.shape)

model = Sequential()

#Adding our first LSTM layer

model.add(LSTM(units = 45, return_sequences = True, input_shape = (X_train.shape[1], 1)))

#Perform some dropout regularization

model.add(Dropout(0.2))

#Adding three more LSTM layers with dropout regularization

for i in  [True, True, False]:

    model.add(LSTM(units = 45, return_sequences = i))

    model.add(Dropout(0.2))

#(Original code for the three additional LSTM layers)

# rnn.add(LSTM(units = 45, return_sequences = True))

# rnn.add(Dropout(0.2))

# rnn.add(LSTM(units = 45, return_sequences = True))

# rnn.add(Dropout(0.2))

# rnn.add(LSTM(units = 45))

# rnn.add(Dropout(0.2))

#Adding our output layer

model.add(Dense(units = 1))##Since we want to output the next day's stock price (a single value), we'll specify units = 1
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 10, batch_size = 32)
results = model.evaluate(X_test,y_test)
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plt.figure(figsize=(12, 8))
Xt = model.predict(X_train)
print('input shape :  ', X_train.shape)
print('state_h shape: ', Xt.shape)
print('result for the first sample/input: \n', Xt[0])
plt.plot(scaler.inverse_transform(X_train.reshape(-1, 1)), label="Actual")
plt.plot(scaler.inverse_transform(Xt.reshape(-1,1)), label="Predicted")
plt.legend()
plt.title("Train Dataset")
#plt.show()

#### Test set

X_test , y_test = create_dataset(test_data, time_step)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Xt = model.predict(X_test)

plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(X_test.reshape(-1, 1)), label="Actual")
plt.plot(scaler.inverse_transform(Xt.reshape(-1,1)), label="Predicted")
plt.legend()
plt.title("Test Dataset")
#plt.show()

#X_train,X_test = X[:int(X.shape[0]*0.90)],X[int(X.shape[0]*0.90):]
#y_train,y_test = y[:int(y.shape[0]*0.90)],y[int(y.shape[0]*0.90):]
#print(X_train.shape[0],X_train.shape[1])
#print(X_test.shape[0], X_test.shape[1])
#print(y_train.shape[0])
#print(y_test.shape[0])
#print("X_train" ,X_train)
#print("y_train" ,y_train)
#print("X_test", X_test)
#print("y_test",y_test)

