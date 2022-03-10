# This is a sample Python



#from pandas import datetime
#from matplotlib import pyplot


#def parser(x):
 #  return pd.to_datetime.strptime('190' + x, '%Y-%m')


#series = pd.read_csv('C:\Master\Sample data.csv',squeeze='true')
#print(series)

#series.plot()
#pyplot.show()

#from pandas import read_csv
#from matplotlib import pyplot
#series = read_csv('C:\Master\Sample data.csv', header=0)
#print(series.head())
#series.plot()
#pyplot.show()
import pandas as pd
#from datetime import datetime
#from matplotlib import pyplot

#"Month","Sales"
#def parser(x):
   #return pd.to_datetime.strptime("190" + x, '%Y-%m')
  # return datetime.datetime.strptime("190" + x, '%Y-%m')
#def parser(x):
 #   return datetime.strptime(date, '%d%b%Y')
#parser = lambda date: datetime.strptime(date, '%d%b%Y')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
#df = pd.read_csv('C:\Master\Automobile.csv', names=['Month','value'], header=0, index_col=0) #header=0, parse_dates=[0], index_col=0)#, squeeze=True) #date_parser=parser)
#print(df)

# Original Series
#fig, axes = plt.subplots(3, 2)
#axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
#plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
#axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
#plot_acf(df.value.diff().dropna(), ax=axes[1, 1])
#plot_pacf(df.value.diff().dropna(), ax=axes[2, 1])
#plt.show()
#from statsmodels.tsa.arima.model import ARIMA

# 1,1,2 ARIMA Model
#model = ARIMA(df.value, order=(1,1,2))
#model_fit = model.fit()
#print(model_fit.summary())
#series.plot()
#pyplot.show()
#x = series.values
#diff = list()
######################3
#for i in range(len(x)):#manual differencing
 #   value = x[i] - x[i - 1]
  #  diff.append(value)
#diff = difference(series.values)#########################
#diff = df.diff() #built in fun for differencing
#pyplot.plot(diff)
#pyplot.show()
#from pmdarima.arima import auto_arima
#ndiffs(df.values, test="adf")  ###########built in fun to determine num of differencing
#df.columns=["Month","Sales"]
#df.head()
#df.describe()
#df.set_index('Month',inplace=True)
#from statsmodels.tsa.stattools import adfuller
#from numpy import log
#test_result=adfuller(df.values)
#print(test_result[0])
#print(test_result[1])
#print(test_result)
####### first way to check stationary data############
#X = df.values
#X = log(X)
#split = round(len(X) / 2)
#X1, X2 = X[0:split], X[split:]
#mean1, mean2 = X1.mean(), X2.mean()
#var1, var2 = X1.var(), X2.var()
#print('mean1=%f, mean2=%f' % (mean1, mean2))
#print('variance1=%f, variance2=%f' % (var1, var2))
###############################################3

###################################
#second way to check stationarity using Augmented Dickey-Fuller test
#X = df.values
#print(X)
#result = adfuller(X)
#if result[1] <= 0.05:
 #       print("P value is less than 0.05 that means we can reject the null hypothesis(Ho). Therefore we can conclude that data has no unit root and is stationary")
  #  else:
   #     print("Weak evidence against null hypothesis that means time series has a unit root which indicates that it is non-stationary ")
#print(result)
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#	print('\t%s: %.3f' % (key, value))
#################################

#compare ARIMA models with different parameters
#####################################################################
import warnings
from math import sqrt
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

#evaluate an ARIMA model for a given order (p,d,q)
#def evaluate_arima_model(X, arima_order):
	# prepare training dataset
#	train_size = int(len(X) * 0.66)
#	train, test = X[0:train_size], X[train_size:]
#	history = [x for x in train]
	# make predictions
#	predictions = list()
#	for t in range(len(test)):
#		model = ARIMA(history, order=arima_order)
#		model_fit = model.fit()
#		yhat = model_fit.forecast()[0]
#		predictions.append(yhat)
#		history.append(test[t])
	 #calculate out of sample error
#	rmse = sqrt(mean_squared_error(test, predictions))
#	return rmse


# evaluate combinations of p, d and q values for an ARIMA model
#def evaluate_models(dataset, p_values, d_values, q_values):
#	dataset = dataset.astype('float32')
#	best_score, best_cfg = float("inf"), None
#	for p in p_values:
#		for d in d_values:
#			for q in q_values:
#				order = (p, d, q)
#				try:
#					rmse = evaluate_arima_model(dataset, order)
#					if rmse < best_score:
#						best_score, best_cfg = rmse, order
#					print('ARIMA%s RMSE=%.3f' % (order, rmse))
#				except:
#					continue
#	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# load dataset
df = read_csv('C:\Master\Automobile.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# evaluate parameters
#p_values = [0, 1, 2, 4]
#d_values = range(0, 3)
#q_values = range(0, 3)
#warnings.filterwarnings("ignore")
#evaluate_models(series.values, p_values, d_values, q_values)
########################################################################################

# Original Series

#split data into train and training set
#train_data, test_data = df[3:int(len(df)*0.9)], df[int(len(df)*0.9):]
#plt.figure(figsize=(10,6))
#plt.grid(True)
#plt.xlabel('Dates')
#plt.ylabel('Closing Prices')
#plt.plot(df_log, 'green', label='Train data')
#plt.plot(test_data, 'blue', label='Test data')
#plt.legend()

#model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
 #                     test='adf',       # use adftest to find optimal 'd'
  #                    max_p=3, max_q=3, # maximum p and q
   #                   m=1,              # frequency of series
    #                  d=None,           # let model determine 'd'
     #                 seasonal=False,   # No Seasonality
      #                start_P=0,
       #               D=0,
        #              trace=True,
         #             error_action='ignore',
          #            suppress_warnings=True,
           #           stepwise=True)
#print(model_autoARIMA.summary())
#model_autoARIMA.plot_diagnostics(figsize=(15,8))


import pyramid as pm
auto_arima_fit = pm.auto_arima(df, start_p=0, start_q=0,
							   test='adf',max_p=3, max_q=3,
							   m=12, seasonal=True,
                             d=0, D=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
print(auto_arima_fit.summary())
auto_arima_fit.plot_diagnostics(figsize=(15,8))

#predictions = model.predict(n_periods=15)
#index_of_fc = np.arange(len(df.value), len(df.value)+n_periods)
#print(index_of_fc)
# predict out of sample
#predictions: Series = pd.Series(model.predict(n_periods=1), index=index_of_fc)


##############LSTM#################

# convert series to supervised learning

#def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    #n_vars = 1 if type(data) is list else data.shape[1]
    #df = DataFrame(data)
    #cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    #for i in range(n_in, 0, -1):
      #  cols.append(df.shift(i))
     #   names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    #for i in range(0, n_out):
        #cols.append(df.shift(-i))
        #if i == 0:
       #     names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
      #  else:
     #       names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    #agg = concat(cols, axis=1)
   # agg.columns = names
    # drop rows with NaN values
  #  if dropnan:
 #       agg.dropna(inplace=True)
#    return agg


# load dataset
#dataset = read_csv('pollution.csv', header=0, index_col=0)
#values = dataset.values
# integer encode direction
#encoder = LabelEncoder()
#values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
#values = values.astype('float32')
# normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)
# frame as supervised learning
#reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
#print(reframed.head())

# split into train and test sets
#values = reframed.values
#n_train_hours = 365 * 24
#train = values[:n_train_hours, :]
#test = values[n_train_hours:, :]
# split into input and outputs
#train_X, train_y = train[:, :-1], train[:, -1]
#test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
#train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
#model = Sequential()
#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#model.add(Dense(1))
#model.compile(loss='mae', optimizer='adam')
# fit network
#history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
 #                   shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

# make a prediction
#yhat = model.predict(test_X)
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
#inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
#test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
#inv_y = inv_y[:, 0]
# calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rmse)
