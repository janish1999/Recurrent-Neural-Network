import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from sklearn.metrics import mean_squared_error


training_set=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=training_set.iloc[:,1:2].values


scaler=MinMaxScaler()
training_set=scaler.fit_transform(training_set)


X_Train=training_set[0:1257]
Y_Train=training_set[1:1258]


X_Train=np.reshape(X_Train,(1257,1,1))


reg=Sequential()

reg.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
reg.add(Dense(units=1))

reg.compile(optimizer='adam',loss='mean_squared_error')


reg.fit(X_Train,Y_Train,batch_size=32,epochs=200)


test_set=pd.read_csv('Google_Stock_Price_Test.csv')
test_set=test_set.iloc[:,1:2].values
Real_Stock_Price=test_set
test_set=scaler.transform(test_set)

test_set=np.reshape(test_set,(20,1,1))


Predicted_Stock_Price=reg.predict(test_set)
Predicted_Stock_Price=scaler.inverse_transform(Predicted_Stock_Price)


plt.plot(Real_Stock_Price,color='red',label='Real Google Stock Price')
plt.plot(Predicted_Stock_Price,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Predictions')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


Rmse=math.sqrt(mean_squared_error(Real_Stock_Price,Predicted_Stock_Price))
print(Rmse)