# Bitcoin Price Predictor

This is a project in COMP809 at Auckland University of Technology attempting to predict bitcoin prices using LR, SVR and a LSTM neural network.

## Purpose

The purpose of this data mining project is to determine, given a dataset tracking the movement of bitcoin over a period of time spanning close to a year a and half, whether it is possible to accurately predict the weighted price of Bitcoin (BTC) on an hourly basis.
By developing three data mining models over the dataset and evaluating their performance by looking at regression metrics, predictions are found and their results can be compared.
The three models that have been developed and evaluated in this project are a Long Short-Term Memory recurrent neural network (LSTM), a Linear Regression (LR) and a Support Vector Regression model.

## Data

The dataset is given in the form of a `CSV` file, containing 10776 data samples with 8 features each.

### The Dataset

The dataset shows the movement of bitcoin spanning over the time period from `01/01/2017` to `25/03/2018`. A section of the raw dataset can be seen below.

| Timestamp       | Open   | High   | Low    | Close  | Volume_BTC  | Volume_Currency | Weighted_Price |
| --------------- | ------ | ------ | ------ | ------ | ----------- | --------------- | -------------- |
| 1/01/2017 10:00 | 961.68 | 963.54 | 960.74 | 962.93 | 77.19857878 | 74300.08902     | 962.8058379    |
| 1/01/2017 11:00 | 962.93 | 964.18 | 962.01 | 963.58 | 174.8056494 | 168584.3686     | 963.265489     |
| 1/01/2017 12:00 | 963.59 | 966.3  | 963.59 | 966.3  | 171.1138657 | 165313.3274     | 964.5229141    |

The goal of this project is predicting the `Weighted_Price` feature given the remaining features.

- `Timestamp` indicates the time of day the remaining features were extracted and represents a one-hour timestep.
- `Open` is the opening price at the time of measurement.
- `High` is the highest price observed in the one-hour timestep.
- `Low` is the lowest price observed in the one-hour timestep.
- `Close` is the closing price observed in the one-hour timestep.
- `Volume_BTC` is the volume of bitcoins present in the bitcoin network at the time of measurement.
- `Volume_Currency` is the volume of actual currency (assumed to be USD) present at the time of measurement.

The `High`, `Low` and `Close` features are unknown at the beginning of the one-hour timestep and are been measured at the end of the timestep. The `Open` feature shows the `Close` feature of the previous timestep since this is known when the one-hour timestep measurement is begun.

### Preprocessing

Given the nature of the dataset, the `High`, `Low` and `Close` features cannot be used in a practical prediction scenario, since these are known "after-the-fact", that is in a realistic prediction setting, knowing what the highest, lowest and closing BTC values would make it trivial to predict the `Weighted_Price`.
A more interesting scenario would be to predict the `Weighted_Price` for a given timestep only knowing the `Open`, `Volume_BTC` and `Volume_Currency` features, that are known at the beginning of each timestep measurement.

The `High`, `Low` and `Close` features have therefore been removed during preprocessing. The `Timestamp` feature has been transformed into simple timestep values from $[0...N]$ where $N = \vert dataset \vert = 10776$.

The dataset is split into training, testing and validation sets. The training dataset is the first 66% of the data samples according to timestamp, with the remaining 33% of the data samples being split into 23% test data for training and 10% validation data after training. The validation data set is used for generating regression metrics to estimate the performance of the generated models.

The dataset is split into subsets according the the timestamps of the data samples in an attempt to "force" the models to improve the prediction accuracy on recent data. The validation dataset consist of the most recent data points and would therefore be more interesting to predict in a real-world scenario, where predicting historical data that is already known is of little value. A model that predicts future values is therefore desireable. Validaing on these recent data samples should hopefully better show the shortcomings of the generated models.

After splitting the dataset, each set has been normalized to values between 0 and 1. Normalisation has been performed after splitting rather than before, in order to in order to improve accuracy and make sure there is no correlation between the values between the data sets.

A section of the processed dataset cat be seen below.

|     | 0                   | 1                   | 2                   | 3                   |
| --- | ------------------- | ------------------- | ------------------- | ------------------- |
| 0   | 0.1565588569028475  | 0.1559126213592233  | 0.1570755441908829  | 0.15641807434215976 |
| 1   | 0.15676235346212766 | 0.15601618122977345 | 0.15728318199208036 | 0.1565236601566244  |
| 2   | 0.1568697996454276  | 0.15635922330097088 | 0.15754150303608974 | 0.1569654961802301  |
| 3   | 0.1572947004612046  | 0.15647087378640778 | 0.15770663231105778 | 0.1570142280945984  |

## Performance of the Models

The LSTM performs slightly better than LR with SVR in last place. The performance of the LR model is very close to the LSTM when looking at the evaluation metrics. The raw metrics after predicting the most recent 10% of the validation data set can be seen below.

|      | MAE         | MSE         | RMSE        | $R^2$       | EV          |
| ---- | ----------- | ----------- | ----------- | ----------- | ----------- |
| LSTM | 0.010710268 | 0.000207735 | 0.014413024 | 0.995107292 | 0.995156721 |
| LR   | 0.011783905 | 0.000216703 | 0.014720851 | 0.994896068 | 0.996982989 |
| SVR  | 0.022348931 | 0.000960744 | 0.030995877 | 0.977371961 | 0.983135368 |

I will only look into the differences between the LSTM and LR models here since they are quite close and will require further examination. The SVR model does clearly not perform as well as the other models and the results will therefore be examined further.

- The mean absolute error is 0.01 higher for the LR model compared to the LSTM model with the mean squared error being 0.00001 higher for the LR model than the LSTM model, indicating a slightly higher accuracy for the LSTM model.
- The root mean squared error is 0.0003 higher for the LR compared to the LSTM, indicating that the LSTM model predicts the outliers of the data set slightly better.
- The $R^2$ score is very close, with the LR having a 0.000217 lower score than the LSTM, indicating that the LSTM model has a slightly better _goodness of fit_.
- The explained variance score for the LR mofel is 0.00182 _higher_ than the score for the LSTM, indicating that the LR model accounts slightly better for the dispersion of the data set.

Given the speed of fitting the LR model compared to the LSTM model the above results are quite impressive.
Training the LSTM with 706 hidden units over 32 epochs takes 6 minutes on the test machine, while the LR model fits in seconds. The LR model looks to be a good choice if quick, relatively accurate results are desired. The improvement gained by training a LSTM might not outweigh the time it takes to train, especially given a bigger data set and/or training over a higher number of epochs.

Plots of the predictions for the three models at two zoom levels can be seen below.

![Plot 1](report/zoom-1.png 'Plot 1')
![Plot 2](report/zoom-2.png 'Plot 2')

The plots show that the SVR model does not perform as well as the LR and LSTM model. Experiments showed that the outlier resistance of the SVR model might prevent it from learning from the price spikes in the input data, since these could be seen as outliers. This is evident in the zoom above.
