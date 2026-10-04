import argparse

import joblib
import pandas as pd
from sklearn import metrics, tree

import config


def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(columns=["kfold", "rainfall"])
    y_train = df_train.rainfall

    x_valid = df_valid.drop(columns=["kfold", "rainfall"])
    y_valid = df_valid.rainfall

    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    f1_score = metrics.f1_score(y_valid, preds)

    print(f"Fold: {fold}, Accuracy: {accuracy}, F1 Score: {f1_score}")

    joblib.dump(clf, f"{config.MODEL_OUTPUT}dt_{fold}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number to run")
    args = parser.parse_args()
    run(args.fold)

    # for fold_ in range(5):
    #     run(fold_)
