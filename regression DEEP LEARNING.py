Import numpy as np
Import pandas as pd
Import matplotlib.pyplot as plt
From sklearn.model_selection import train_test_split
From sklearn.metrics import mean_squared_error, r2_score
From sklearn.datasets import fetch_california_housing
From tensorflow.keras.models import sequential
From tensorflow.keras.layers import dense
From tensorflow.keras.optimizers import adam
# load the california housing dataset
Data = fetch_california_housing()
X = pd.dataframe(data.data, columns=data.feature_names)
Y = pd.series(data.target)
# display the first few rows of the dataset
Print(x.head())	
Print(y.head())
# split data into training and test sets
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# normalize the data (deep learning models often perform better with normalized data)
From sklearn.preprocessing import standardscaler
Scaler = standardscaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
# initialize the deep learning model (sequential model)
Model = sequential()
# add input layer and first hidden layer
Model.add(dense(64, activation='relu', input_shape=(x_train.shape[1],)))
# add second hidden layer
Model.add(dense(64, activation='relu'))
# add output layer (single neuron for regression output)
Model.add(dense(1))
# compile the model
Model.compile(optimizer=adam(learning_rate=0.001), loss='mean_squared_error')
# train the model
History = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)
# make predictions on the test set
Y_pred = model.predict(x_test).flatten()
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
Plt.title('predicted vs actual prices (neural network)')
Plt.xlabel('actual prices')
Plt.ylabel('predicted prices')
Plt.grid()
Plt.show()
# plot the training and validation loss over epochs
Plt.figure(figsize=(10, 6))
Plt.plot(history.history['loss'], label='training loss')
Plt.plot(history.history['val_loss'], label='validation loss')
Plt.title('model loss over epochs')
Plt.xlabel('epochs')
Plt.ylabel('loss (mean squared error)')
Plt.legend()
Plt.grid()
Plt.show()
