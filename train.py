import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# Read
data = read_csv("D1.csv", header=0, index_col=0, skiprows=14)
values = data.values
values = values.astype('float64')

# Scale inputs
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
values = scaled

# Train on the first year of data
split = int(values.shape[0]*0.1)
train = values[:split*8, :]
test = values[split*8:split*9, :]
val = values[split*9:split*10, :]

# Split data into test and training
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]

# Reshape input to [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

# Training shapes
print("Training shapes:")
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Instantiate model
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

# Fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=24,
                    validation_data=(test_X, test_y), shuffle=False)

# Plot loss history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

# Make a prediction
predict = model.predict(val_X, batch_size=24)

# Plot actual and prediction
plt.plot(predict[:, -1], label='predict')
plt.plot(val_y, label='actual')
plt.legend()
plt.show()
