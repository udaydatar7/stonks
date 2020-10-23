import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras

print("Data is available on the following stocks:\n AAPL\n BA\n T\n MGM\n AMZN\n IBM\n TSLA\n GOOG\n sp500\n ")
stock_name = input("Enter stock name to predict:")

stock_price_df = pd.read_csv(r'C:\Users\udayd\Downloads/stock (2).csv')
stock_vol_df = pd.read_csv(r'C:\Users\udayd\Downloads/stock_volume.csv')
stock_price_df = stock_price_df.sort_values(by = ['Date'])
stock_vol_df = stock_vol_df.sort_values(by = ['Date'])
stock_price_df.isnull().sum()
stock_vol_df.isnull().sum()
stock_vol_df.isnull().sum()
stock_price_df.info()
stock_vol_df.info()
stock_vol_df.describe()

def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()

interactive_plot(stock_price_df, 'Stock Prices')

def individual_stock(price_df, vol_df, name):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name], 'Volume': vol_df[name]})

def trading_window(data):
  
  # 1 day window 
  n = 1

  # Create a column containing the prices for the next 1 days
  data['Target'] = data[['Close']].shift(-n)
  
  # return the new dataset 
  return data
price_volume_df = individual_stock(stock_price_df, stock_vol_df, f'{stock_name}')
price_volume_target_df = trading_window(price_volume_df)
price_volume_target_df = price_volume_target_df[:-1]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns = ['Date']))
price_volume_target_scaled_df
price_volume_target_scaled_df.shape
X = price_volume_target_scaled_df[:,:2]
y = price_volume_target_scaled_df[:,2:]
X.shape, y.shape
split = int(0.65 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]
X_train.shape, y_train.shape
X_test.shape, y_test.shape
def show_plot(data, title):
  plt.figure(figsize = (13, 5))
  plt.plot(data, linewidth = 3)
  plt.title(title)
  plt.grid()

show_plot(X_train, 'Training Data')
show_plot(X_test, 'Testing Data')


price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'sp500')   
training_data = price_volume_df.iloc[:, 1:3].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_data)
X = []
y = []
for i in range(1, len(price_volume_df)):
    X.append(training_set_scaled [i-1:i, 0])
    y.append(training_set_scaled [i, 0])

X = np.asarray(X)
y = np.asarray(y)
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences= True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()

history = model.fit(
    X_train, y_train,
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
)
predicted = model.predict(X)
test_predicted = []

for i in predicted:
  test_predicted.append(i[0])
df_predicted = price_volume_df[1:][['Date']]
df_predicted['predictions'] = test_predicted
close = []
for i in training_set_scaled:
  close.append(i[0])
df_predicted['Close'] = close[1:]
df_predicted
interactive_plot(df_predicted, "Original Vs Prediction")
