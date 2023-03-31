import sys
import matplotlib

matplotlib.use('Qt5Agg')


from matplotlib.pyplot import plot, scatter, savefig, show, xlabel, ylabel
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

boston_data = pd.read_csv("boston.csv")
print(boston_data.head(2))  # will show the first two rows table

sns.boxplot(data=boston_data)  # will plot boxgraph
corr = boston_data.corr() #determine the correlation between different variables
sns.heatmap(corr, annot=True)  # will provide the heatmap
sns.kdeplot(data=boston_data, x='medv')  # will show the density map for the given x ,and or y
show()  # to display the graph

x1 = boston_data[['nox']]
y1 = boston_data['medv']

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.33, random_state=5)
"""The line test_size=0.33 suggests that
 the test data should be 33% of the dataset 
 and the rest should be train data"""

regressor = lm()
regressor.fit(x_train, y_train)
""""" a function to measure 
 how well a machine learning model generalizes to similar data
 to that on which it was trained"""
y_pred = regressor.predict(x_test)  # predict the label of a new set of data
boston_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(y_test)
print(y_pred)
xlabel('True values')  # creates label
ylabel('Predictions')
scatter(x1, y1)  # creates scatter plot graph

# Mean Absolute Error
error = mean_absolute_error(x_test, y_pred)
print("Mean absolute error : " + str(error))

# Mean Square Error
ms = mean_squared_error(x_test, y_pred)
print(ms)

# Root Mean Square Error
rms = np.sqrt(ms)
print(rms)

# r2 Score
r2_result = r2_score(x_test, y_pred)
print(r2_result)

x1 = np.array(x_test['nox'])
y1 = np.array(y_pred)
scatter(x1, y1)
m = np.polyfit(x1, y1, 1)
#print(m) # to get the constant values

plot(x_test['nox'], -34.21048323 * x_test['nox'] + 41.43617943) # using formula y=mx+c
plot(x_test['nox'], y_pred, c='r') # linear reg with color c= read
xlabel("Nitric Oxide Content")
ylabel("Median Value")

sns.pairplot(boston_data, x_vars=['nox', 'rm', 'dis', 'ptratio', 'lstat'], y_vars='medv', height=9, aspect=0.6,
             kind='reg')

show()
savefig(sys.stdout.buffer)
sys.stdout.flush()