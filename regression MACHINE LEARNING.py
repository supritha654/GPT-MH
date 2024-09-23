Import numpy as np
Import pandas as pd
Import matplotlib.pyplot as plt
From sklearn.model_selection import train_test_split
From sklearn.linear_model import linearregression
From sklearn.metrics import mean_squared_error, r2_score
From sklearn.datasets import fetch_california_housing
# load the california housing dataset
Data = fetch_california_housing()
X = pd.dataframe(data.data, columns=data.feature_names)
Y = pd.series(data.target)
# display the first few rows of the dataset
Print(x.head())
Print(y.head())
# split data into training and test sets
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# initialize the regression model
Model = linearregression()
# train the model
Model.fit(x_train, y_train)
# make predictions on the test set
Y_pred = model.predict(x_test)
# evaluate the model
Mse = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)
# output the results
Print(f"mean squared error: {mse}")
Print(f"accuracy: {r2}")
# visualize the predictions vs actual values
Plt.figure(figsize=(10, 6))
Plt.scatter(y_test, y_pred, alpha=0.5)
Plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # ideal line
Plt.title('predicted vs actual prices')
Plt.xlabel('actual prices')
Plt.ylabel('predicted prices')
Plt.grid()
Plt.show()
