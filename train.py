import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import mlflow
import mlflow.sklearn

# -----------------------
# Config
# -----------------------
DATA_PATH = "data/train.csv"
MODEL_PATH = "models/churn_model.pkl"
EXPERIMENT_NAME = "churn-experiment"

os.makedirs("models", exist_ok=True)

# -----------------------
# MLflow setup
# -----------------------
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(DATA_PATH)

    X = df[["age", "salary"]]
    y = df["churn"]

    # -----------------------
    # Train-test split
    # -----------------------
    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # -----------------------
    # Model
    # -----------------------
    C_value = 1.0
    model = LogisticRegression(C=C_value, max_iter=200)
    model.fit(X_train, y_train)

    # -----------------------
    # Evaluation
    # -----------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # -----------------------
    # MLflow logging
    # -----------------------
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", C_value)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.log_model(model, artifact_path="model")

    # -----------------------
    # Save model for Flask/DVC
    # -----------------------
    joblib.dump(model, MODEL_PATH)

    print("✅ Training complete")
    print(f"Accuracy: {accuracy}")

