# Question 5: Machine Learning Demand Forecasting using sampledataset.csv

import os
import math

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


GRAPH_FOLDER = "../Graphs"
DATASET_PATH = "../Dataset/Q5_sampledataset.csv"

os.makedirs(GRAPH_FOLDER, exist_ok=True)


def load_sales_data():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "dataset was not found. "
            "Place it in the main project folder, outside the Code folder."
        )

    data = pd.read_csv(DATASET_PATH)
    return data


def save_rmse_comparison(results_df):

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["RMSE"])
    plt.title("RMSE Comparison")
    plt.xlabel("Model")
    plt.ylabel("Root Mean Squared Error")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Question 5-plot1.png"), dpi=300)
    plt.close()


def save_actual_vs_predicted(y_test, predictions, model_name):

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions)
    plt.title(f"Actual vs Predicted Demand: {model_name}")
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_FOLDER, "Question 5-plot2.png"), dpi=300)
    plt.close()


if __name__ == "__main__":

    sales_data = load_sales_data()

    print("\nSample Sales Dataset:")
    print(sales_data.head())

    features = sales_data[
        [
            "month",
            "price",
            "promotion",
            "website_visits",
            "previous_month_sales"
        ]
    ]

    target = sales_data["demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = []
    predictions_store = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        })

        predictions_store[model_name] = predictions

    results_df = pd.DataFrame(results)

    print("\nQuestion 5 Model Results:")
    print(results_df)

    best_model_name = results_df.sort_values("RMSE").iloc[0]["Model"]
    best_predictions = predictions_store[best_model_name]

    print(f"\nBest Model Based on RMSE: {best_model_name}")

    save_rmse_comparison(results_df)
    save_actual_vs_predicted(y_test, best_predictions, best_model_name)

    print("\nGraphs saved in the Graphs folder.")