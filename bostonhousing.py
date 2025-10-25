from statistics import LinearRegression
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)


#Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
#Train the model
model.fit(X_train, y_train)


#Predict with testing data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print(f'Mean Squared Error: {mse}')
