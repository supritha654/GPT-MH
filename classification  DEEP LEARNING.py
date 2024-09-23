Import numpy as np
Import pandas as pd
Import matplotlib.pyplot as plt
From sklearn.datasets import fetch_california_housing
From sklearn.model_selection import train_test_split
From sklearn.preprocessing import standardscaler, labelbinarizer
From sklearn.preprocessing import kbinsdiscretizer
Import tensorflow as tf
From tensorflow.keras.models import sequential
From tensorflow.keras.layers import dense, dropout
# load the dataset
Housing = fetch_california_housing()
Df = pd.dataframe(housing.data, columns=housing.feature_names)
Df['target'] = housing.target  # adding the target variable
# discretize the target variable into 3 classes (low, medium, high house prices)
Binner = kbinsdiscretizer(n_bins=3, encode='ordinal', strategy='quantile')  # you can also try 'uniform'
Df['target_binned'] = binner.fit_transform(df[['target']])
# define features and target
X = df[housing.feature_names]
Y = df['target_binned']
# split the data into training and testing sets
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# feature scaling
Scaler = standardscaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
# one-hot encode the target variable
Lb = labelbinarizer()
Y_train_encoded = lb.fit_transform(y_train)
Y_test_encoded = lb.transform(y_test)
# build the deep learning model
Model = sequential([
    dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),  # input layer
    dropout(0.3),  # dropout to prevent overfitting
    dense(32, activation='relu'),  # hidden layer
    dense(16, activation='relu'),  # hidden layer
    dense(3, activation='softmax')  # output layer (3 classes for classification)])
# compile the model
Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# train the model
History = model.fit(x_train_scaled, y_train_encoded, epochs=20, batch_size=32, validation_split=0.2)
# evaluate the model
Test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test_encoded)
Print(f'test accuracy: {test_accuracy}')
# plot training history
Plt.plot(history.history['accuracy'], label='training accuracy')
Plt.plot(history.history['val_accuracy'], label='validation accuracy')
Plt.xlabel('epochs')
Plt.ylabel('accuracy')
Plt.legend()
Plt.title('training and validation accuracy')
Plt.show() 
