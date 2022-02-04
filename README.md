# Stock Price Prediction Documentation

This Repository contains the source code for Predicting Stock Prices using Machine Learning Algorithms such as (Linear/Ridge/Lasso/SVR Regression).

<!-- ## Introduction:

Accurately estimating the value of real estate is an important problem for many
stakeholders including house owners, house buyers, agents, creditors, and investors. It is also a difficult one. Though it is common knowledge that factors
such as the size, number of rooms and location affect the price, there are many
other things at play. Additionally, prices are sensitive to changes in market
demand and the peculiarities of each situation, such as when a property needs
to be urgently sold. The sales price of a property can be predicted in various ways, but is often
based on regression techniques. All regression techniques essentially involve
one or more predictor variables as input and a single target variable as output. -->

## Dataset Source:
- [Yahoo Finance](https://finance.yahoo.com/)

## Prerequisites:
Install yfinance
`pip install yfinance`

## Tool Prerequisites:
1. Pandas
2. Matplotlib
3. Scikit-Learn (train_test_split, LinearRegression, mean_squared_error, r2_score, Lasso, Ridge, SVR, GridSearchCV)
4. NumPy
5. Seaborn
6. yfinance - `import yfinance as yf`

<!-- ## Linear Regression:

[Linear regression](https://en.wikipedia.org/wiki/Linear_regression) is a linear approach for modelling the relationship between a Dependant variable(Output) and an Independant variable(Input). Different techniques can be used to prepare or train the linear regression equation from data, the most common of which is called [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares#:~:text=In%20statistics%2C%20ordinary%20least%20squares,in%20a%20linear%20regression%20model.&text=Under%20these%20conditions%2C%20the%20method,the%20errors%20have%20finite%20variances.).
 -->
## Steps to create the Model:
1. Import the required libraries and modules,
 ```
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import yfinance as yf
```
2. Using the Stock code, download the dataset using `yf.download()` function
3. Create a dataframe of the given dataset using Pandas.
4. Plot the dataframe using `sns.displot` function and observe the curves.
5. Split the data into Training and Validating datasets.
6. Use the `LinearRegression()` function in the variable `lr` and Fit the data using the `model.fit()` function.
7. Find the predicted values using the `model.predict()` function and assign it to `pred1`.
8. compare `pred1` to `y_val` and calculate the [Cost Function](https://www.analyticsvidhya.com/blog/2021/03/data-science-101-introduction-to-cost-function/) using `calc_metrics` function.

## Additional Regression Algorithms:

1. Ridge Regression - `Ridge().fit()` function
2. Lasso Regression - `Lasso().fit()` function

## Validation:
Validate the predicted values by calculating the MSE and RMSE values,
```
def calc_metrics(actual, predicted):
  mse = mean_squared_error(actual, predicted)
  rmse = np.sqrt(mse)
  r2s = r2_score(actual, predicted)

  print("MSE:", mse)
  print("RMSE:", rmse)
  print("r2_score:", r2s)
  
calc_metrics(y_val, pred1)
```
## Visualization:
#### Visualize the data by using Matplotlib,
```
data.Close.plot(figsize=(10, 7), color="b")

plt.ylabel("{} Price".format(stocks))
plt.xlabel("{} Interval".format(stocks))
plt.title("{} Chart".format(stocks))
plt.show
```
#### Visualize the data by using Seaborn,
```
sns.displot(data["High"], kind="kde")
sns.displot(data["Low"], kind="kde")
```
