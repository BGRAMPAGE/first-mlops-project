from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score

import mlflow
import mlflow.sklearn

mlflow.set_experiment("House Price Prediction")

import joblib
house = fetch_california_housing()

X = house.data

y = house.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# model = LinearRegression()

# model.fit(X_train,y_train)

# predictions = model.predict(X_test)

# rmse = mean_squared_error(y_test, predictions) ** 0.5
# r2 = r2_score(y_test, predictions)
# ac = accuracy_score(y_test,predictions)
# print(f"RMSE: {rmse:.4f}")
# print(f"R² Score: {r2:.4f}")

from sklearn.ensemble import RandomForestRegressor

# model2 = RandomForestRegressor(n_estimators=40,criterion='absolute_error',random_state=42)

# model2.fit(X_train,y_train)

# pred = model2.predict(X_test)

# rmse = mean_squared_error(y_test, pred) ** 0.5
# r2 = r2_score(y_test, pred)


# print(f"RMSE: {rmse:.4f}")
# print(f"R² Score: {r2:.4f}")

# joblib.dump(model2, "house_price_model.pkl")

# print("Model saved successfully!")

