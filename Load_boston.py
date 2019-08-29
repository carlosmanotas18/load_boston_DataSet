
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

boston = load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

model=LinearRegression()
regression = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("ِAccuracy Score : " , explained_variance_score(y_test, y_pred)*100 , " %")
print("Mean absolute Error :" , mean_absolute_error(y_test, y_pred, multioutput='raw_values'))
print("Mean squared Error :" , mean_squared_error(y_test, y_pred))

