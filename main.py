import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# load & read data from pandas datafrome
df = pd.read_csv('data.csv')

# defining features & target variables
x = df['bedrooms', 'sqft', 'location']
y = df['price']

# split data into training & testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# create & train model
lr = LinearRegression()
lr.fit(x_train, y_train)

# predicting with the model
y_pred = lr.predict(x_test)

# saving the predictions to a file
np.savetxt('predictions.csv', y_pred, delimiter=',')

# evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error:', mse)
print('R2 score:', r2)
