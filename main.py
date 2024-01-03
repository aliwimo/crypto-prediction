"""Main module that is used to test package"""

# import dependencies
import pandas as pd
import matplotlib.pyplot as plt

# import sklearn models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# import dataset
df = pd.read_csv('datasets/XRP-USD.csv')

# convert date column to date format
df['Date'] = pd.to_datetime(df['Date'])

# Shift data by one day to use previous day's values for training
df[['Open', 'High', 'Low']] = df[['Open', 'High', 'Low']].shift(0)

# drop rows with NaN values resulting from the shift
df = df.dropna()

# split dataset into training and test subsets
train_mask = (df['Date'] >= '2022-8-10') & (df['Date'] <= '2022-12-20')
train_df = df.loc[train_mask]
test_mask = (df['Date'] >= '2022-12-21') & (df['Date'] <= '2023-2-8')
test_df = df.loc[test_mask]

# split each of training and test subsets into inputs (X) and outputs (Y)
X_train = train_df[['Open', 'High', 'Low']]
y_train = train_df['Close']
date_train = train_df['Date']
X_test = test_df[['Open', 'High', 'Low']]
y_test = test_df['Close']
date_test = test_df['Date']

# create model to train
model = LinearRegression()

# fit data into model
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

# print results of the model
train_score = r2_score(y_train, y_fit)
test_score = r2_score(y_test, y_pred)
print(f'Training set r2 score: {train_score}\nTest set r2 score: {test_score}')

# plot model graph
plt.clf()
ax = plt.axes()
ax.grid(linestyle=':', linewidth=0.5, alpha=1, zorder=1)
plt.ylabel("BTC Price ($)")
line = [None, None, None, None]
line[0], = ax.plot(date_train, y_train, linestyle=':', color='black', linewidth=0.7, zorder=2, label='Targeted')
line[1], = ax.plot(date_train, y_fit, linestyle='-', color='red', linewidth=0.7, zorder=3, label='Trained')
line[2], = ax.plot(date_test, y_test, linestyle=':', color='black', linewidth=0.7, zorder=2)
line[3], = ax.plot(date_test, y_pred, linestyle='-', color='blue', linewidth=0.7, zorder=3, label='Predicted')
plt.axvline(x=date_test.iloc[0], linestyle='-', color='black', linewidth='1')
plt.draw()
plt.legend()
plt.show()
