import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from src.models.ModelTrainer import ModelTrainer

def main():
    trainset = pd.read_csv("data/trainset.csv")
    testset = pd.read_csv("data/testset.csv")

    X_train, y_train = trainset.iloc[:, :-1], trainset["y"]
    X_test, y_test = testset.iloc[:, :-1], testset["y"]

    model_params = {"solver": "lbfgs", "max_iter": 1}

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Binary-problem-test")
    with mlflow.start_run():
        mlflow.log_params(model_params)

        trainer = ModelTrainer(model=LogisticRegression(**model_params))
        fitted_model = trainer.train(X_train, y_train)

        mlflow.sklearn.log_model(fitted_model, artifact_path="model")

        train_score = ModelTrainer.score(fitted_model, X_train, y_train)
        test_score = ModelTrainer.score(fitted_model, X_test, y_test)

        for m_name, m_score in train_score.items():
            mlflow.log_metric(f"train_{m_name}", m_score)
        for m_name, m_score in test_score.items():
            mlflow.log_metric(f"test_{m_name}", m_score)
        

if __name__ == "__main__":
    main()
