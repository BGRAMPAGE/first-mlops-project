import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
import joblib
import matplotlib.pyplot as plt

mlflow.set_experiment("House Price Prediction")

house = fetch_california_housing()

X = house.data

y = house.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


from sklearn.ensemble import RandomForestRegressor

for trees in [50, 100, 200]:

    with mlflow.start_run():

        model = RandomForestRegressor(
            n_estimators=trees,
            criterion="absolute_error",
            random_state=42
        )

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", trees)
        mlflow.log_param("criterion", "absolute_error")
        mlflow.log_param("random_state", 42)


        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        feature_names = house.feature_names
        importances = model.feature_importances_

        plt.figure(figsize=(8,5))
        plt.bar(feature_names, importances)
        plt.xticks(rotation=45)
        plt.title(f"Feature Importance ({trees} Trees)")
        plt.tight_layout()

        plot_name = f"feature_importance_{trees}.png"
        plt.savefig(plot_name)
        mlflow.log_artifact(plot_name)
        plt.close()
        
        report_name = f"training_report_{trees}.txt"

        with open(report_name, "w") as f:
            f.write(f"Model: RandomForest\n")
            f.write(f"Trees: {trees}\n")
            f.write(f"Criterion: absolute_error\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R2: {r2:.4f}\n")

        mlflow.sklearn.log_model(model, "house_price_model")
        mlflow.log_artifact(report_name)
        mlflow.sklearn.log_model(sk_model=model,name="house_price_model",registered_model_name="HousePriceModel (RandomForestRegressor)")

        print(f"Trees: {trees}")
        print(f"RMSE: {rmse:.4f}")
        
        print(f"R²: {r2:.4f}")
        print("-" * 30)