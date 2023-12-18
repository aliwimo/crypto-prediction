"""Main module that is used to test package"""

# pylint: disable=unused-import
# import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import sklearn metrics and utilities
# from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# import sklearn models
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import tree


from sklearn.metrics import r2_score

# import dataset
df = pd.read_csv('datasets/XRP-USD.csv')

# convert date column to date format
df['Date'] = pd.to_datetime(df['Date'])

# split dataset into training and test subsets
train_mask = (df['Date'] >= '2022-8-10') & (df['Date'] <= '2022-12-20')
train_df = df.loc[train_mask]
test_mask = (df['Date'] >= '2022-12-21') & (df['Date'] <= '2023-2-8')
test_df = df.loc[test_mask]

# print(test_mask)

# split each of training and test subsets into inputs (X) and outputs (Y)
X_train = train_df[['Open', 'High', 'Low']]
y_train = train_df['Close']
date_train = train_df['Date']
X_test = test_df[['Open', 'High', 'Low']]
y_test = test_df['Close']
date_test = test_df['Date']

# convert dataframes to numpy objects
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()



# choose a model to train
# model = MLPRegressor(max_iter=50000, hidden_layer_sizes=(4, 4, 4, ))
# model = MLPRegressor(random_state=1, max_iter=500)
# model = tree.DecisionTreeRegressor()
model = LinearRegression()
# model = KNeighborsRegressor(n_neighbors=2)
# model = make_pipeline(StandardScaler(), SVR(C=100.0, coef0=1.0, kernel='poly', max_iter=50000))


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
