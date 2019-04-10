# Bitcoin-Price-Predictor-COMP809
A project in COMP809 at Auckland University of Technology attempting to predict bitcoin prices using a LSTM RNN.

The dataset (found in D1.csv) has the format:

| Timestamp       | Open   | High   | Low    | Close  | Volume_BTC  | Volume_Currency | Weighted_Price |
|-----------------|--------|--------|--------|--------|-------------|-----------------|----------------|
| 1/01/2017 10:00 | 961.68 | 963.54 | 960.74 | 962.93 | 77.19857878 | 74300.08902     | 962.8058379    |
| 1/01/2017 11:00 | 962.93 | 964.18 | 962.01 | 963.58 | 174.8056494 | 168584.3686     | 963.265489     |
| 1/01/2017 12:00 | 963.59 | 966.3  | 963.59 | 966.3  | 171.1138657 | 165313.3274     | 964.5229141    |

The goal of the project is to train an LSTM RNN to predict bitcoin prices (_Weighted_Price_ above). 
