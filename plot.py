import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import preprocessing


def normalize(data):
    np_data = np.array(data).reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(np_data)
    return scaler.transform(np_data)


day = []
opens = []
high = []
low = []
close = []
volume_btc = []
volume_currency = []
weighted_price = []

with open('D1.csv', 'r') as csvfile:
    rows = csv.reader(csvfile, delimiter=',')
    next(rows, None)
    day_number = 1
    for row in rows:
        day.append(day_number)
        day_number = day_number + 1
        opens.append(float(row[1]))
        high.append(float(row[2]))
        low.append(float(row[3]))
        close.append(float(row[4]))
        volume_btc.append(float(row[5]))
        volume_currency.append(float(row[6]))
        weighted_price.append(float(row[7]))

plt.subplot(3, 1, 1)
plt.plot(day, opens)
plt.plot(day, high)
plt.plot(day, low)
plt.plot(day, close)
plt.plot(day, weighted_price)
plt.xlabel('day')
plt.ylabel('price')

plt.subplot(3, 1, 2)
plt.plot(day, volume_btc)
plt.xlabel('day')
plt.ylabel('BTC volume')

plt.subplot(3, 1, 3)
plt.plot(day, volume_currency)
plt.xlabel('day')
plt.ylabel('Currency volume')
plt.show()
