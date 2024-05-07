########################################################################################################################
# IMPORTS
########################################################################################################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

########################################################################################################################
# IMPLEMENT AN OLS  PARAMETER ESTIMATION FOR LINEAR REGRESSION USING ONLY NUMPY
########################################################################################################################
'''
In this exercise we want you to visualise and understand linear regression.

data:
-----
winequality-red.csv
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
we want to estimate the quality as labelled by "expert-drinkers" based on only ONE feature (alcohol)

what you DONT need to care about:
---------------------------------
- missing data
- categorical/binary features

output:
-------
- use sklearn.linear_model to predict quality based on alcohol content
- plot alcohol content vs the quality predictions with red x-symbols
- plot alcohol content vs the true quality with blue filled circles
- draw a green line between predicted qualities and true qualities to visualise the residuals (= Size of the error we made)
- the result should look like this https://i.stack.imgur.com/zoYKG.png (But with different data!!)
- then label the x-axis and y-axis
- Calculate the Mean Squared Error (MSE)
'''

# read the data --------------------------------------------------
# -- predefined --
pth = r'winequality-red.csv'
df = pd.read_csv(pth, sep=";")

X = df['alcohol'].values.reshape(-1, 1)
y = df['quality'].values

# Create and fit the linear regression model
model = lr().fit(X, y)

# Predict the quality using the model
y_pred = model.predict(X)

# Set the size of the plot
plt.figure(figsize=(8, 5))

# Plot the true quality vs the alcohol content
plt.scatter(X, y, color='blue', marker='o', label='True Quality')

# Plot the predicted quality vs the alcohol content
plt.scatter(X, y_pred, color='red', marker='x', label='Predicted Quality')

# Draw a line between the predicted and true quality
for i in range(len(X)):
    # Label the first line to create a legend
    if i == 0:
        plt.plot([X[i], X[i]], [y[i], y_pred[i]], color='green', label='Residuen')
    else:
        plt.plot([X[i], X[i]], [y[i], y_pred[i]], color='green')

# Draw the regression line
plt.plot(X, y_pred, color='red', label='Regression Line')

# Label the Diagram
plt.xlabel('Alcohol Content')
plt.ylabel('Quality')
plt.title('Alcohol Content vs Quality')
plt.legend(loc='lower right')

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()

# Calculate the Mean Squared Error
mse = mean_squared_error(y, y_pred)
print('Mean Squared Error (MSE):', mse)
