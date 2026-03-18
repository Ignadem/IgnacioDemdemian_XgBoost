from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "clean_train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "clean_test.csv"


def main():
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    X_train = train_data.drop(["Outcome"], axis=1)
    y_train = train_data["Outcome"]
    X_test = test_data.drop(["Outcome"], axis=1)
    y_test = test_data["Outcome"]

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.001,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Predicciones:")
    print(y_pred)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
